"use strict";

// --- tab switching ---
document.querySelectorAll("nav button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "documents") loadDocuments();
    if (btn.dataset.tab === "stats") loadStats();
  });
});

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

// --- chat ---
let sessionId = null;
const messagesEl = document.getElementById("messages");

function addMessage(role, html) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = html;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

const STATUS_LABEL = {
  grounded: "Terverifikasi",
  out_of_context: "Di luar konteks",
  hallucinated: "Halusinasi",
};

function renderCitations(citations) {
  if (!citations || !citations.length) return "";
  let html = '<div class="citations"><strong>Sitasi:</strong>';
  for (const c of citations) {
    const cls =
      c.status === "grounded" ? (c.verified ? "ok" : "warn")
      : c.status === "out_of_context" ? "warn"
      : "bad";
    html += '<div class="cite ' + cls + '">';
    html += '<span class="badge">' + escapeHtml(STATUS_LABEL[c.status] || c.status) + "</span>";
    html += "<strong>" + escapeHtml(c.label || "Pasal " + c.pasal) + "</strong>";
    if (c.text) html += "<p>" + escapeHtml(c.text) + "</p>";
    html += "</div>";
  }
  return html + "</div>";
}

async function sendQuestion() {
  const input = document.getElementById("question");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  addMessage("user", escapeHtml(q));
  const pending = addMessage("bot", "<em>Memproses...</em>");
  try {
    const res = await fetch("/chat/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, session_id: sessionId }),
    });
    const data = await res.json();
    if (!res.ok) {
      pending.innerHTML = '<span class="error">Gagal: ' + escapeHtml(data.detail || res.statusText) + "</span>";
      return;
    }
    sessionId = data.session_id;
    let html = escapeHtml(data.answer).replace(/\n/g, "<br>");
    html += renderCitations(data.citations);
    const notes = data.verification && data.verification.notes;
    if (notes && notes.length) {
      html += '<div class="notes"><strong>Catatan verifikasi:</strong><ul>';
      for (const n of notes) html += "<li>" + escapeHtml(n) + "</li>";
      html += "</ul></div>";
    }
    html += '<div class="confidence">Tingkat keyakinan: <strong>'
      + Math.round(data.confidence * 100) + "%</strong></div>";
    html += '<div class="disclaimer">' + escapeHtml(data.disclaimer) + "</div>";
    pending.innerHTML = html;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (e) {
    pending.innerHTML = '<span class="error">Gagal terhubung ke server.</span>';
  }
}

document.getElementById("send").addEventListener("click", sendQuestion);
document.getElementById("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

document.querySelectorAll(".example").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("question").value = btn.textContent;
    sendQuestion();
  });
});

// --- documents ---
async function loadDocuments() {
  const el = document.getElementById("doc-list");
  el.innerHTML = "<em>Memuat...</em>";
  try {
    const data = await (await fetch("/documents/list")).json();
    if (!data.documents.length) {
      el.innerHTML = "<p>Belum ada dokumen.</p>";
      return;
    }
    el.innerHTML = "";
    for (const d of data.documents) {
      const div = document.createElement("div");
      div.className = "doc";
      div.innerHTML =
        "<div><strong>" + escapeHtml(d.short_name) + "</strong> — " + escapeHtml(d.law_name) +
        "<br><small>" + escapeHtml(d.law_type) + " · " + d.chunk_count + " pasal · " +
        escapeHtml(d.filename) + "</small></div>";
      const btn = document.createElement("button");
      btn.textContent = "Hapus";
      btn.addEventListener("click", async () => {
        if (!confirm("Hapus dokumen ini?")) return;
        await fetch("/documents/" + d.id, { method: "DELETE" });
        loadDocuments();
      });
      div.appendChild(btn);
      el.appendChild(div);
    }
  } catch (e) {
    el.innerHTML = '<span class="error">Gagal memuat dokumen.</span>';
  }
}

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("file");
  if (!fileInput.files.length) return;
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  for (const f of ["short_name", "law_name", "law_type", "law_number"]) {
    const v = document.getElementById(f).value.trim();
    if (v) fd.append(f, v);
  }
  const year = document.getElementById("law_year").value;
  if (year) fd.append("law_year", year);

  const btn = e.target.querySelector("button");
  btn.disabled = true;
  btn.textContent = "Mengunggah...";
  try {
    const res = await fetch("/documents/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      alert("Gagal: " + (data.detail || "kesalahan"));
    } else {
      alert(
        data.already_existed
          ? "Dokumen sudah pernah diunggah."
          : "Berhasil diunggah: " + data.document.chunk_count + " pasal."
      );
      e.target.reset();
      loadDocuments();
    }
  } catch (err) {
    alert("Gagal terhubung ke server.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Unggah";
  }
});

// --- stats ---
let typeChart = null;
async function loadStats() {
  try {
    const data = await (await fetch("/stats")).json();
    document.getElementById("stat-cards").innerHTML =
      '<div class="card"><span>' + data.total_documents + "</span>Dokumen</div>" +
      '<div class="card"><span>' + data.total_chunks + "</span>Pasal</div>" +
      '<div class="card"><span>' + data.total_queries + "</span>Pertanyaan</div>";

    const labels = data.documents_by_type.map((x) => x.law_type);
    const counts = data.documents_by_type.map((x) => x.count);
    if (typeChart) typeChart.destroy();
    typeChart = new Chart(document.getElementById("type-chart"), {
      type: "doughnut",
      data: {
        labels: labels.length ? labels : ["Belum ada data"],
        datasets: [
          {
            data: counts.length ? counts : [1],
            backgroundColor: ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#6b7280"],
          },
        ],
      },
      options: { plugins: { title: { display: true, text: "Dokumen per Jenis" } } },
    });
  } catch (e) {
    document.getElementById("stat-cards").innerHTML =
      '<span class="error">Gagal memuat statistik.</span>';
  }
}
