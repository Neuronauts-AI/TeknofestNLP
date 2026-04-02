
    const report = document.getElementById("report");
    const historyBox = document.getElementById("history");
    const ollamaStyleBox = document.getElementById("ollama-style-box");
    const sampleSelect = document.getElementById("sample-select");
    const alertsBox = document.getElementById("alerts-box");
    const semanticQueryEl = document.getElementById("semantic-query");
    const semanticResultsEl = document.getElementById("semantic-results");
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

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderSemanticSearch(data) {
      if (!data) {
        semanticResultsEl.innerHTML = '<div class="tiny">Henüz semantik arama yapılmadı.</div>';
        return;
      }
      if (!data.available) {
        semanticResultsEl.innerHTML = `<div class="tiny">Semantik arama kullanılamıyor: ${data.error || 'bilinmeyen hata'}</div>`;
        return;
      }
      const items = data.results || [];
      if (!items.length) {
        semanticResultsEl.innerHTML = '<div class="tiny">Benzer vaka bulunamadı.</div>';
        return;
      }
      semanticResultsEl.innerHTML = items.map(item => `
        <div class="history-item">
          <strong>Vaka ${item.row}</strong>
          <div class="tiny">Skor: ${item.score}</div>
          <div class="tiny"><strong>Bulgular</strong></div>
          <div class="tiny">${escapeHtml(item.findings || "")}</div>
          ${item.impression ? `<div class="tiny" style="margin-top:8px;"><strong>Sonuç</strong></div><div class="tiny">${escapeHtml(item.impression || "")}</div>` : ""}
        </div>
      `).join('');
    }

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
          <div class="tiny">${(item.report || '').replaceAll('\n', '<br>')}</div>
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
      renderSemanticSearch(data.semantic_search);
      semanticQueryEl.value = data.semantic_search?.query || semanticQueryEl.value;
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
      renderSemanticSearch(data.semantic_search);
      semanticQueryEl.value = data.semantic_search?.query || semanticQueryEl.value;
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
      renderSemanticSearch(data.semantic_search);
      semanticQueryEl.value = data.semantic_search?.query || semanticQueryEl.value;
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
    document.getElementById("semantic-search-button").addEventListener("click", async () => {
      try {
        const response = await fetch("/semantic-search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: semanticQueryEl.value, top_k: 5 })
        });
        if (!response.ok) {
          const errorText = await response.text();
          semanticResultsEl.innerHTML = `<div class="tiny">Semantik arama hatası: ${escapeHtml(errorText)}</div>`;
          return;
        }
        const data = await response.json();
        renderSemanticSearch(data);
      } catch (error) {
        semanticResultsEl.innerHTML = `<div class="tiny">Semantik arama hatası: ${escapeHtml(String(error))}</div>`;
      }
    });
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
  