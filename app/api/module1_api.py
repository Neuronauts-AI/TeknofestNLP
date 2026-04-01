import csv
import json
import random
import tempfile
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.module1.asr_whisper import DEFAULT_ASR_MODEL, transcribe_audio_file
from app.module1.transcript_sections import parse_transcript_sections
from app.module2.ollama_style_client import (
    DEFAULT_MODEL as DEFAULT_OLLAMA_MODEL,
    classify_report_style_with_ollama,
)
from app.module2.ollama_critical_alerts import classify_critical_alerts


app = FastAPI(title="Module 1 - Report Generation API", version="0.5.0")
DATASET_PATH = Path("data/processed/module1/all.csv")
REPORT_HISTORY_LIMIT = 20
report_history: deque[dict] = deque(maxlen=REPORT_HISTORY_LIMIT)


class OllamaStylePayload(BaseModel):
    available: bool
    model: str
    label: str
    score: float
    reason: str
    error: str = ""


class ReportRequest(BaseModel):
    findings: str = ""
    impression: str = ""


class ReportResponse(BaseModel):
    report: str
    ollama_style: OllamaStylePayload
    critical_alerts: dict


class AsrResponse(BaseModel):
    text: str
    findings: str
    impression: str
    report: str
    model_id: str
    backend: str
    ollama_style: OllamaStylePayload
    critical_alerts: dict


class HistoryItem(BaseModel):
    findings: str
    impression: str
    report: str
    ollama_style: OllamaStylePayload
    critical_alerts: dict


class HistoryResponse(BaseModel):
    items: list[HistoryItem]


def build_report(findings: str, impression: str) -> str:
    sections = []
    findings = findings.strip()
    impression = impression.strip()

    if findings:
        sections.append(f"Bulgular:\n{findings}")
    if impression:
        sections.append(f"Sonuç:\n{impression}")

    return "\n\n".join(sections)


def load_sample_reports(limit: int = 5) -> list[dict[str, str]]:
    if not DATASET_PATH.exists():
        return []

    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        all_samples = [
            {
                "id": str(index),
                "findings_tr": row.get("findings_tr", ""),
                "impression_tr": row.get("impression_tr", ""),
                "report_tr": row.get("report_tr", ""),
            }
            for index, row in enumerate(reader, start=1)
        ]

    if len(all_samples) <= limit:
        return all_samples

    return random.sample(all_samples, k=limit)


def build_ollama_payload(findings: str, impression: str) -> OllamaStylePayload:
    ollama_result = classify_report_style_with_ollama(
        findings=findings,
        impression=impression,
        model=DEFAULT_OLLAMA_MODEL,
    )
    return OllamaStylePayload(
        available=ollama_result.available,
        model=ollama_result.model,
        label=ollama_result.label,
        score=ollama_result.score,
        reason=ollama_result.reason,
        error=ollama_result.error,
    )


def build_alerts_payload(report: str) -> dict:
    result = classify_critical_alerts(report, model=DEFAULT_OLLAMA_MODEL)
    return {
        "available": result.available,
        "model": result.model,
        "alerts": [
            {
                "finding": item.finding,
                "severity": item.severity,
                "status": item.status,
                "reason": item.reason,
            }
            for item in result.alerts
        ],
        "error": result.error,
    }


def save_history(findings: str, impression: str, report: str, ollama_style: OllamaStylePayload, critical_alerts: dict) -> None:
    report_history.appendleft(
        {
            "findings": findings,
            "impression": impression,
            "report": report,
            "ollama_style": ollama_style.model_dump(),
            "critical_alerts": critical_alerts,
        }
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Modül 1 Demo</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: #fffdf8;
      --text: #1d1a16;
      --muted: #6f655a;
      --line: #d8ccbb;
      --accent: #165e63;
      --accent-dark: #10474b;
      --soft: #efe5d7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top right, rgba(22, 94, 99, 0.08), transparent 30%),
        linear-gradient(180deg, #f0e7d9 0%, var(--bg) 100%);
      color: var(--text);
    }
    .wrap {
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 20px 52px;
    }
    .hero {
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      justify-content: space-between;
      align-items: end;
      margin-bottom: 24px;
    }
    .eyebrow {
      font-size: 13px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 10px;
      font-weight: 700;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(32px, 5vw, 58px);
      line-height: 0.94;
    }
    .sub {
      max-width: 720px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.6;
      margin: 0;
    }
    .badge {
      padding: 10px 14px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--accent-dark);
      font-size: 14px;
      font-weight: 700;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: 0 10px 28px rgba(60, 40, 20, 0.06);
    }
    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .field + .field { margin-top: 16px; }
    label {
      display: block;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 8px;
      color: var(--accent-dark);
    }
    textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      font: inherit;
      font-size: 16px;
      line-height: 1.55;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 170px;
      resize: vertical;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 13px 18px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #ece3d3;
      color: var(--accent-dark);
    }
    button:hover { background: var(--accent-dark); }
    button.secondary:hover { background: #dfd1bb; }
    button.recording {
      background: #8f2116;
      color: #fff;
    }
    button.recording:hover {
      background: #73180f;
    }
    .report {
      white-space: pre-wrap;
      line-height: 1.7;
      font-size: 17px;
      min-height: 220px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .history-list {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .history-item {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: #fff;
    }
    .pill {
      display: inline-block;
      margin-bottom: 8px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .pill-good {
      background: #e4efe5;
      color: #235b31;
    }
    .pill-bad {
      background: #f8d7d2;
      color: #8f2116;
    }
    .pill-warn {
      background: #f3ead0;
      color: #8a6720;
    }
    .filter-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
      margin-bottom: 4px;
    }
    .filter-chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: #fff;
      color: var(--accent-dark);
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }
    .filter-chip.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .history-item strong {
      display: block;
      margin-bottom: 6px;
      color: var(--accent-dark);
    }
    .tiny {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.5;
    }
    @media (max-width: 920px) {
      .grid, .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <div class="eyebrow">Türkçe Klinik NLP Platformu</div>
        <h1>Modül 1<br>Rapor Üretim Demo Katmanı</h1>
        <p class="sub">Metinden rapora çekirdek akış, örnek veri yükleme, rapor geçmişi ve yalnızca Ollama tabanlı gerçeklik değerlendirmesi tek ekranda.</p>
      </div>
      <div class="badge" id="stream-status">Canlı akış: bağlanıyor</div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="split">
          <div class="field">
            <label for="sample-select">Örnek Veri</label>
            <select id="sample-select">
              <option value="">Elle gir</option>
            </select>
          </div>
          <div class="field">
            <label>Canlı Durum</label>
            <div class="muted" id="live-status">WebSocket bağlantısı kuruluyor...</div>
          </div>
        </div>

        <div class="field">
          <label for="findings">Bulgular</label>
          <textarea id="findings"></textarea>
        </div>
        <div class="field">
          <label for="impression">Sonuç</label>
          <textarea id="impression"></textarea>
        </div>

        <div class="field">
          <label for="audio-file">Sesli Girdi</label>
          <div>
            <input id="audio-file" type="file" accept="audio/*">
          </div>
          <div class="actions">
            <button class="secondary" id="transcribe-audio">Sesi Yazıya Çevir</button>
            <button class="secondary" id="start-recording">Mikrofonu Başlat</button>
            <button class="secondary" id="stop-recording" disabled>Kaydı Durdur</button>
          </div>
          <div class="muted" id="audio-status">Whisper large-v3 ham transkripti çıkaracak. Bulgular ve Sonuç ayrımı için konuşmada başlıkları açık söylemelisin: örneğin “Bulgular ... Sonuç ...”.</div>
        </div>

        <div class="actions">
          <button id="generate">Rapor Oluştur</button>
          <button class="secondary" id="load-samples">Örnekleri Yenile</button>
          <button class="secondary" id="send-live">Canlı Akışa Gönder</button>
        </div>
      </div>

      <div class="card">
        <label>Oluşan Rapor</label>
        <div id="report" class="report">Burada rapor gösterilecek.</div>
        <div class="muted" style="margin-top:14px;">API dokümanı için <a href="/docs">/docs</a> adresini kullanabilirsin.</div>
      </div>
    </div>

    <div class="card" style="margin-top:20px;">
      <label>Ollama Sınıflandırması</label>
      <div id="ollama-style-box" class="history-list">
        <div class="tiny">Henüz Ollama sınıflandırması yapılmadı.</div>
      </div>
    </div>

    <div class="card" style="margin-top:20px;">
      <label>Canlı Uyarılar</label>
      <div class="filter-row">
        <button type="button" class="filter-chip active" data-alert-filter="all">Tümü</button>
        <button type="button" class="filter-chip" data-alert-filter="present">Mevcut</button>
        <button type="button" class="filter-chip" data-alert-filter="uncertain">Belirsiz</button>
        <button type="button" class="filter-chip" data-alert-filter="absent">Yok</button>
      </div>
      <div id="alerts-box" class="history-list">
        <div class="tiny">Henüz kritik uyarı üretilmedi.</div>
      </div>
    </div>

    <div class="card" style="margin-top:20px;">
      <label>Son Üretilen Rapor Geçmişi</label>
      <div id="history" class="history-list">
        <div class="tiny">Henüz rapor geçmişi yok.</div>
      </div>
    </div>
  </div>

  <script>
    const report = document.getElementById("report");
    const historyBox = document.getElementById("history");
    const ollamaStyleBox = document.getElementById("ollama-style-box");
    const sampleSelect = document.getElementById("sample-select");
    const alertsBox = document.getElementById("alerts-box");
    const alertFilterButtons = Array.from(document.querySelectorAll("[data-alert-filter]"));
    const findingsEl = document.getElementById("findings");
    const impressionEl = document.getElementById("impression");
    const audioFileEl = document.getElementById("audio-file");
    const audioStatusEl = document.getElementById("audio-status");
    const startRecordingButton = document.getElementById("start-recording");
    const stopRecordingButton = document.getElementById("stop-recording");
    const liveStatus = document.getElementById("live-status");
    const streamStatus = document.getElementById("stream-status");
    let samples = [];
    let socket = null;
    let activeAlertFilter = "all";
    let latestAlertsPayload = null;
    let mediaRecorder = null;
    let recordedChunks = [];

    function getAlertStatusLabel(status) {
      if (status === "present") return "mevcut";
      if (status === "uncertain") return "belirsiz";
      if (status === "absent") return "yok";
      return status || "-";
    }

    function renderHistory(items) {
      if (!items.length) {
        historyBox.innerHTML = '<div class="tiny">Henüz rapor geçmişi yok.</div>';
        return;
      }
      historyBox.innerHTML = items.map((item, index) => `
        <div class="history-item">
          <strong>Kayıt ${index + 1}</strong>
          <div class="tiny">${(item.report || '').replaceAll('\\n', '<br>')}</div>
          <div class="tiny" style="margin-top:8px;">Ollama: ${(item.ollama_style || {}).label || 'yok'} | Skor: ${(item.ollama_style || {}).score ?? '-'}</div>
          <div class="tiny">Uyarı sayısı: ${((item.critical_alerts || {}).alerts || []).length}</div>
        </div>
      `).join('');
    }

    function renderOllamaStyle(data) {
      if (!data) {
        ollamaStyleBox.innerHTML = '<div class="tiny">Henüz Ollama sınıflandırması yapılmadı.</div>';
        return;
      }
      if (!data.available) {
        ollamaStyleBox.innerHTML = `
          <div class="history-item">
            <strong>Ollama kullanılamıyor</strong>
            <div class="tiny">Model: ${data.model || '-'}</div>
            <div class="tiny">${data.reason || 'Bağlantı kurulamadı.'}</div>
            <div class="tiny">${data.error || ''}</div>
          </div>
        `;
        return;
      }
      const pillClass = data.label === "uygun" ? "pill-good" : "pill-bad";
      ollamaStyleBox.innerHTML = `
        <div class="history-item">
          <div class="pill ${pillClass}">${data.label}</div>
          <strong>Ollama / ${data.model}</strong>
          <div class="tiny">Skor: ${data.score}</div>
          <div class="tiny">${data.reason || ''}</div>
        </div>
      `;
    }

    function renderAlerts(data) {
      latestAlertsPayload = data;
      if (!data) {
        alertsBox.innerHTML = '<div class="tiny">Henüz kritik uyarı üretilmedi.</div>';
        return;
      }
      if (!data.available) {
        alertsBox.innerHTML = `
          <div class="history-item">
            <strong>Uyarı sistemi kullanılamıyor</strong>
            <div class="tiny">${data.error || ''}</div>
          </div>
        `;
        return;
      }
      const priority = { present: 0, uncertain: 1, absent: 2 };
      const items = (data.alerts || [])
        .slice()
        .sort((a, b) => {
          const statusDiff = (priority[a.status] ?? 9) - (priority[b.status] ?? 9);
          if (statusDiff !== 0) {
            return statusDiff;
          }
          return String(a.finding || "").localeCompare(String(b.finding || ""), "tr");
        });
      const filteredItems = activeAlertFilter === "all"
        ? items
        : items.filter(item => item.status === activeAlertFilter);
      if (!items.length) {
        alertsBox.innerHTML = '<div class="tiny">Kritik bulgu uyarısı üretilmedi.</div>';
        return;
      }
      if (!filteredItems.length) {
        alertsBox.innerHTML = '<div class="tiny">Seçili filtrede uyarı yok.</div>';
        return;
      }
      alertsBox.innerHTML = filteredItems.map(item => {
        const pillClass =
          item.status === 'present' ? 'pill-bad' :
          item.status === 'uncertain' ? 'pill-warn' :
          'pill-good';
        return `
          <div class="history-item">
            <div class="pill ${pillClass}">${getAlertStatusLabel(item.status)}</div>
            <strong>${item.finding}</strong>
            <div class="tiny">Önem: ${item.severity}</div>
            <div class="tiny">${item.reason || ''}</div>
          </div>
        `;
      }).join('');
    }

    function setAlertFilter(filterValue) {
      activeAlertFilter = filterValue;
      alertFilterButtons.forEach(button => {
        button.classList.toggle("active", button.dataset.alertFilter === filterValue);
      });
      renderAlerts(latestAlertsPayload);
    }

    async function loadSamples() {
      const response = await fetch("/samples");
      const data = await response.json();
      samples = data.samples || [];
      sampleSelect.innerHTML = '<option value="">Elle gir</option>' + samples.map(sample =>
        `<option value="${sample.id}">Örnek ${sample.id}</option>`
      ).join('');
    }

    async function loadHistory() {
      const response = await fetch("/history");
      const data = await response.json();
      renderHistory(data.items || []);
    }

    async function generateReport() {
      report.textContent = "Rapor oluşturuluyor...";
      const response = await fetch("/generate-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          findings: findingsEl.value,
          impression: impressionEl.value
        })
      });

      if (!response.ok) {
        report.textContent = "Bir hata oluştu.";
        return;
      }

      const data = await response.json();
      report.textContent = data.report || "Boş rapor döndü.";
      renderOllamaStyle(data.ollama_style);
      renderAlerts(data.critical_alerts);
      await loadHistory();
    }

    async function transcribeAudio() {
      const file = audioFileEl.files && audioFileEl.files[0];
      if (!file) {
        audioStatusEl.textContent = "Önce bir ses dosyası seç.";
        return;
      }

      audioStatusEl.textContent = "Ses dosyası Whisper ile çözümleniyor...";
      const formData = new FormData();
      formData.append("audio", file);

      const response = await fetch("/transcribe-audio", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        audioStatusEl.textContent = `ASR hatası: ${errorText}`;
        return;
      }

      const data = await response.json();
      findingsEl.value = data.findings || "";
      impressionEl.value = data.impression || "";
      report.textContent = data.report || "Boş rapor döndü.";
      renderOllamaStyle(data.ollama_style);
      renderAlerts(data.critical_alerts);
      await loadHistory();
      audioStatusEl.textContent = `ASR tamamlandı (${data.model_id}) ve konuşmadaki başlıklara göre alanlar dolduruldu.`;
    }

    async function transcribeAudioBlob(blob, filename) {
      audioStatusEl.textContent = "Mikrofon kaydı Whisper ile çözümleniyor...";
      const formData = new FormData();
      formData.append("audio", blob, filename);

      const response = await fetch("/transcribe-audio", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        audioStatusEl.textContent = `ASR hatası: ${errorText}`;
        return;
      }

      const data = await response.json();
      findingsEl.value = data.findings || "";
      impressionEl.value = data.impression || "";
      report.textContent = data.report || "Boş rapor döndü.";
      renderOllamaStyle(data.ollama_style);
      renderAlerts(data.critical_alerts);
      await loadHistory();
      audioStatusEl.textContent = `Mikrofon kaydı çözüldü (${data.model_id}) ve konuşmadaki başlıklara göre alanlar dolduruldu.`;
    }

    async function startRecording() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        audioStatusEl.textContent = "Bu tarayıcı mikrofon erişimini desteklemiyor.";
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const preferredMimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "";
        mediaRecorder = preferredMimeType
          ? new MediaRecorder(stream, { mimeType: preferredMimeType })
          : new MediaRecorder(stream);
        recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            recordedChunks.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const mimeType = mediaRecorder.mimeType || "audio/webm";
          const extension = mimeType.includes("ogg") ? "ogg" : "webm";
          const blob = new Blob(recordedChunks, { type: mimeType });
          const tracks = mediaRecorder.stream ? mediaRecorder.stream.getTracks() : [];
          tracks.forEach(track => track.stop());
          startRecordingButton.disabled = false;
          stopRecordingButton.disabled = true;
          startRecordingButton.classList.remove("recording");
          await transcribeAudioBlob(blob, `microphone.${extension}`);
        };

        mediaRecorder.start();
        startRecordingButton.disabled = true;
        stopRecordingButton.disabled = false;
        startRecordingButton.classList.add("recording");
        audioStatusEl.textContent = "Mikrofon kaydı başladı...";
      } catch (error) {
        audioStatusEl.textContent = `Mikrofon hatası: ${error}`;
      }
    }

    function stopRecording() {
      if (!mediaRecorder || mediaRecorder.state === "inactive") {
        return;
      }
      audioStatusEl.textContent = "Kayıt durduruluyor...";
      mediaRecorder.stop();
    }

    function connectSocket() {
      socket = new WebSocket(`ws://${location.host}/ws/report-stream`);
      socket.onopen = () => {
        liveStatus.textContent = "Canlı akış bağlandı.";
        streamStatus.textContent = "Canlı akış: aktif";
      };
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "report_preview") {
          report.textContent = payload.report || "";
          renderOllamaStyle(payload.ollama_style);
          renderAlerts(payload.critical_alerts);
        }
      };
      socket.onclose = () => {
        liveStatus.textContent = "Canlı akış bağlantısı kapandı.";
        streamStatus.textContent = "Canlı akış: kapalı";
      };
    }

    sampleSelect.addEventListener("change", () => {
      const selected = samples.find(item => item.id === sampleSelect.value);
      if (!selected) return;
      findingsEl.value = selected.findings_tr || "";
      impressionEl.value = selected.impression_tr || "";
      report.textContent = selected.report_tr || "Örnek seçildi.";
    });

    document.getElementById("generate").addEventListener("click", generateReport);
    document.getElementById("load-samples").addEventListener("click", loadSamples);
    document.getElementById("transcribe-audio").addEventListener("click", transcribeAudio);
    startRecordingButton.addEventListener("click", startRecording);
    stopRecordingButton.addEventListener("click", stopRecording);
    document.getElementById("send-live").addEventListener("click", () => {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        liveStatus.textContent = "WebSocket bağlantısı hazır değil.";
        return;
      }
      socket.send(JSON.stringify({
        findings: findingsEl.value,
        impression: impressionEl.value
      }));
    });

    alertFilterButtons.forEach(button => {
      button.addEventListener("click", () => setAlertFilter(button.dataset.alertFilter || "all"));
    });

    loadSamples();
    loadHistory();
    connectSocket();
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/samples")
def samples() -> dict[str, list[dict[str, str]]]:
    return {"samples": load_sample_reports()}


@app.post("/transcribe-audio", response_model=AsrResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
) -> AsrResponse:
    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await audio.read())

    try:
        asr_result = transcribe_audio_file(temp_path, model_id=DEFAULT_ASR_MODEL)
        parsed_sections = parse_transcript_sections(asr_result.text)
        findings = parsed_sections.findings.strip()
        impression = parsed_sections.impression.strip()
        report = build_report(findings, impression)
        ollama_style = build_ollama_payload(findings, impression)
        critical_alerts = build_alerts_payload(report)
        save_history(findings, impression, report, ollama_style, critical_alerts)
        return AsrResponse(
            text=asr_result.text,
            findings=findings,
            impression=impression,
            report=report,
            model_id=asr_result.model_id,
            backend=asr_result.backend,
            ollama_style=ollama_style,
            critical_alerts=critical_alerts,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/history", response_model=HistoryResponse)
def history() -> HistoryResponse:
    return HistoryResponse(items=[HistoryItem(**item) for item in report_history])


@app.post("/generate-report", response_model=ReportResponse)
def generate_report(payload: ReportRequest) -> ReportResponse:
    report = build_report(payload.findings, payload.impression)
    ollama_style = build_ollama_payload(payload.findings, payload.impression)
    critical_alerts = build_alerts_payload(report)
    save_history(payload.findings, payload.impression, report, ollama_style, critical_alerts)
    return ReportResponse(report=report, ollama_style=ollama_style, critical_alerts=critical_alerts)


@app.websocket("/ws/report-stream")
async def report_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)
            findings = payload.get("findings", "")
            impression = payload.get("impression", "")
            report = build_report(findings, impression)
            ollama_style = build_ollama_payload(findings, impression)
            critical_alerts = build_alerts_payload(report)
            await websocket.send_json(
                {
                    "type": "report_preview",
                    "report": report,
                    "ollama_style": ollama_style.model_dump(),
                    "critical_alerts": critical_alerts,
                }
            )
    except WebSocketDisconnect:
        return
