/* Dr. Asha — frontend logic: SSE streaming chat, clarify chips, profile drawer, uploads. */
"use strict";

document.addEventListener("DOMContentLoaded", () => {
    // ------------------------------------------------------------------
    // Backend URL resolution:
    //  - served BY the backend (same origin, e.g. http://127.0.0.1:8000) → use it
    //  - frontend hosted separately on localhost (Live Server :5500, etc.) → local backend
    //  - anywhere else → the deployed backend, unless window.APP_BACKEND_URL overrides it
    // ------------------------------------------------------------------
    const isLocalHost = location.hostname === "localhost" || location.hostname === "127.0.0.1";
    const servedByBackend = isLocalHost && location.port === "8000";
    const BACKEND_URL =
        window.APP_BACKEND_URL ||
        (servedByBackend
            ? location.origin
            : isLocalHost
                ? "http://127.0.0.1:8000"
                : "https://document-chatbot-kambaa.onrender.com");

    // ------------------------------------------------------------------
    // Elements
    // ------------------------------------------------------------------
    const $ = (id) => document.getElementById(id);
    const chatWindow = $("chat-window");
    const composer = $("composer");
    const userInput = $("user-input");
    const sendButton = $("send-button");
    const attachButton = $("attach-button");
    const fileInput = $("file-input");
    const newChatButton = $("new-chat-button");
    const themeToggle = $("theme-toggle");
    const profileButton = $("profile-button");
    const profileDot = $("profile-dot");
    const presenceLine = $("presence-line");
    const dragOverlay = $("drag-overlay");
    const welcomeCard = $("welcome-card");
    const toast = $("toast");

    const drawer = $("profile-drawer");
    const drawerBackdrop = $("drawer-backdrop");
    const drawerClose = $("drawer-close");
    const profileSave = $("profile-save");
    const drawerFiles = $("drawer-files");

    const pf = {
        name: $("pf-name"), age: $("pf-age"), sex: $("pf-sex"), pregnant: $("pf-pregnant"),
        diet: $("pf-diet"),
        conditions: $("pf-conditions"), medications: $("pf-medications"),
        allergies: $("pf-allergies"), preferences: $("pf-preferences"),
    };

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    const sessionId = (() => {
        let id = sessionStorage.getItem("asha-session");
        if (!id) {
            id = crypto.randomUUID();
            sessionStorage.setItem("asha-session", id);
        }
        return id;
    })();

    let profile = {};
    let uploadedFiles = [];
    let busy = false;          // awaiting/streaming an answer
    let uploading = false;
    let activeController = null;

    // Transcript persistence (per tab) — survives page reloads within the session
    const CHAT_KEY = "asha-chat";
    let chatLog = [];
    try { chatLog = JSON.parse(sessionStorage.getItem(CHAT_KEY) || "[]"); } catch (e) { chatLog = []; }
    const saveChatLog = () => {
        try { sessionStorage.setItem(CHAT_KEY, JSON.stringify(chatLog.slice(-40))); } catch (e) {}
    };

    function restoreChat() {
        if (!chatLog.length) return;
        welcomeCard?.remove();
        for (const entry of chatLog) {
            if (entry.u) addMessage("user", null).textContent = entry.u;
            if (entry.b) {
                const bubble = addMessage("bot", null);
                if (entry.err) {
                    bubble.textContent = entry.b;
                    bubble.style.background = "var(--danger-bg)";
                } else {
                    bubble.innerHTML = markdown(entry.b);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------
    const markdown = (text) =>
        DOMPurify.sanitize(marked.parse(text || "", { gfm: true, breaks: true }), {
            ADD_ATTR: ["target"],
        });

    function toastMsg(message, isError = false) {
        toast.textContent = message;
        toast.classList.toggle("error", isError);
        toast.hidden = false;
        clearTimeout(toastMsg._t);
        toastMsg._t = setTimeout(() => { toast.hidden = true; }, 3500);
    }

    function scrollDown() { chatWindow.scrollTop = chatWindow.scrollHeight; }

    function setBusy(state) {
        busy = state;
        sendButton.disabled = state || uploading || !userInput.value.trim();
        attachButton.disabled = state || uploading;
        userInput.disabled = state || uploading;
    }

    function setUploading(state) {
        uploading = state;
        sendButton.disabled = state || busy || !userInput.value.trim();
        attachButton.disabled = state || busy;
        userInput.disabled = state || busy;
    }

    function addMessage(sender, html) {
        welcomeCard?.remove();
        const wrapper = document.createElement("div");
        wrapper.className = `message-wrapper ${sender}-message`;
        const avatarSvg = sender === "bot"
            ? '<img src="assets/logo.svg" alt="">'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>';
        wrapper.innerHTML = `<div class="avatar ${sender}" aria-hidden="true">${avatarSvg}</div>`;
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        if (html !== null && html !== undefined) bubble.innerHTML = html;
        wrapper.appendChild(bubble);
        chatWindow.appendChild(wrapper);
        scrollDown();
        return bubble;
    }

    function addStatusLine(text) {
        const el = document.createElement("div");
        el.className = "status-line";
        el.innerHTML = `<span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span> <span></span>`;
        el.querySelector("span:last-child").textContent = text;
        chatWindow.appendChild(el);
        scrollDown();
        return el;
    }

    // ------------------------------------------------------------------
    // Profile drawer
    // ------------------------------------------------------------------
    function profileHasData(p) {
        return p && Object.values(p).some((v) =>
            Array.isArray(v) ? v.length > 0 : (v !== null && v !== "" && v !== undefined));
    }

    function renderProfileDot() { profileDot.hidden = !profileHasData(profile); }

    function profileToForm(p) {
        p = p || {};
        pf.name.value = p.name || "";
        pf.age.value = p.age ?? "";
        pf.sex.value = p.sex || "";
        pf.pregnant.checked = p.pregnant === true;
        pf.diet.value = p.diet || "";
        pf.conditions.value = (p.conditions || []).join(", ");
        pf.medications.value = (p.medications || []).join(", ");
        pf.allergies.value = (p.allergies || []).join(", ");
        const pref = (p.preferences || [])[0] || "";
        [...pf.preferences.options].forEach((o) => { if (o.value === pref) pf.preferences.value = pref; });
        drawerFiles.textContent = uploadedFiles.length
            ? `Reports in this session: ${uploadedFiles.join(", ")}`
            : "No reports uploaded in this session yet.";
    }

    function formToProfileUpdates() {
        const split = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);
        const updates = {};
        if (pf.name.value.trim()) updates.name = pf.name.value.trim();
        if (pf.age.value) updates.age = parseInt(pf.age.value, 10);
        if (pf.sex.value) updates.sex = pf.sex.value;
        if (pf.pregnant.checked) updates.pregnant = true;
        if (pf.diet.value) updates.diet = pf.diet.value;
        if (pf.conditions.value.trim()) updates.conditions = split(pf.conditions.value);
        if (pf.medications.value.trim()) updates.medications = split(pf.medications.value);
        if (pf.allergies.value.trim()) updates.allergies = split(pf.allergies.value);
        if (pf.preferences.value) updates.preferences = [pf.preferences.value];
        return updates;
    }

    function openDrawer() { profileToForm(profile); drawer.hidden = false; drawerBackdrop.hidden = false; drawerClose.focus(); }
    function closeDrawer() { drawer.hidden = true; drawerBackdrop.hidden = true; profileButton.focus(); }

    profileButton.addEventListener("click", openDrawer);
    drawerClose.addEventListener("click", closeDrawer);
    drawerBackdrop.addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !drawer.hidden) closeDrawer();
    });

    profileSave.addEventListener("click", async () => {
        try {
            const res = await fetch(`${BACKEND_URL}/api/profile`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, profile: formToProfileUpdates() }),
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            profile = data.profile;
            renderProfileDot();
            toastMsg("Profile saved — Dr. Asha will use this from now on.");
            closeDrawer();
        } catch (err) {
            console.error(err);
            toastMsg("Could not save your profile. Please try again.", true);
        }
    });

    function applyProfile(p) {
        if (p && typeof p === "object") { profile = p; renderProfileDot(); }
    }

    // ------------------------------------------------------------------
    // Theme
    // ------------------------------------------------------------------
    themeToggle.addEventListener("click", () => {
        const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        try { localStorage.setItem("theme", next); } catch (e) {}
    });

    // ------------------------------------------------------------------
    // Health line + session hydration (after a page reload the backend may
    // still hold this session's profile and uploaded reports)
    // ------------------------------------------------------------------
    (async () => {
        try {
            const res = await fetch(`${BACKEND_URL}/api/health`);
            const h = await res.json();
            presenceLine.textContent = `Personalized answers · powered by ${h.model}`;
        } catch (e) { /* keep default */ }
        try {
            const res = await fetch(`${BACKEND_URL}/api/profile?session_id=${encodeURIComponent(sessionId)}`);
            if (res.ok) {
                const data = await res.json();
                applyProfile(data.profile);
                if (Array.isArray(data.files)) uploadedFiles = data.files;
            }
        } catch (e) { /* fresh session or backend down — nothing to hydrate */ }
    })();

    // ------------------------------------------------------------------
    // Clarify UI
    // ------------------------------------------------------------------
    function renderClarify(preamble, questions) {
        welcomeCard?.remove();
        const wrapper = document.createElement("div");
        wrapper.className = "message-wrapper bot-message";
        wrapper.innerHTML = `<div class="avatar bot" aria-hidden="true"><img src="assets/logo.svg" alt=""></div>`;
        const box = document.createElement("div");
        box.className = "bubble";
        box.innerHTML = `<div class="clarify-box"><p class="clarify-preamble"></p><div class="clarify-body"></div></div>`;
        box.querySelector(".clarify-preamble").textContent = preamble || "A couple of quick things so I can tailor this for you:";
        wrapper.appendChild(box);
        chatWindow.appendChild(wrapper);

        const body = box.querySelector(".clarify-body");
        const answers = new Map();
        let index = 0;

        const askOne = () => {
            body.innerHTML = "";
            if (index >= questions.length) { submitAnswers(); return; }
            const q = questions[index];

            const qEl = document.createElement("p");
            qEl.className = "clarify-question";
            qEl.textContent = q.question;
            body.appendChild(qEl);

            if (q.type === "choice" && Array.isArray(q.options) && q.options.length) {
                const row = document.createElement("div");
                row.className = "chip-row";
                q.options.forEach((opt) => {
                    const b = document.createElement("button");
                    b.type = "button";
                    b.className = "chip";
                    b.textContent = opt;
                    b.addEventListener("click", () => {
                        answers.set(q.id || q.question, opt);
                        index += 1;
                        askOne();
                    });
                    row.appendChild(b);
                });
                body.appendChild(row);
            } else {
                const inputRow = document.createElement("div");
                inputRow.className = "clarify-input-row";
                const input = document.createElement("input");
                input.type = "text";
                input.placeholder = "Type your answer…";
                const go = document.createElement("button");
                go.type = "button"; go.className = "chip";
                go.textContent = "OK";
                const commit = () => {
                    if (!input.value.trim()) return;
                    answers.set(q.id || q.question, input.value.trim());
                    index += 1;
                    askOne();
                };
                go.addEventListener("click", commit);
                input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); commit(); } });
                inputRow.append(input, go);
                body.appendChild(inputRow);
                setTimeout(() => input.focus(), 30);
            }

            if (q.optional) {
                const skip = document.createElement("p");
                skip.className = "skip-note";
                skip.innerHTML = `Rather not say? <button type="button">Skip this</button>`;
                skip.querySelector("button").addEventListener("click", () => { index += 1; askOne(); });
                body.appendChild(skip);
            }
        };

        const submitAnswers = () => {
            const reply = [...answers.entries()]
                .map(([k, v], i) => `${i + 1}) ${v}`)
                .join("  ");
            wrapper.remove();
            sendUserMessage(reply || "I'd rather not share that — please do your best with what you have.");
        };

        askOne();
        scrollDown();
    }

    // ------------------------------------------------------------------
    // Chat streaming (SSE over fetch)
    // ------------------------------------------------------------------
    async function sendUserMessage(text) {
        if (busy || uploading || !text.trim()) return;
        addMessage("user", null).textContent = text;
        const logEntry = { u: text, b: null };
        chatLog.push(logEntry);
        saveChatLog();
        setBusy(true);
        activeController = new AbortController();

        let statusEl = null;
        let answerBubble = null;
        let answerText = "";

        const renderAnswer = () => {
            if (!answerBubble) return;
            answerBubble.innerHTML = markdown(answerText);
            answerBubble.classList.add("streaming");
            scrollDown();
        };
        let renderQueued = false;
        const scheduleRender = () => {
            if (renderQueued) return;
            renderQueued = true;
            requestAnimationFrame(() => { renderQueued = false; renderAnswer(); });
        };

        const handleEvent = (ev) => {
            switch (ev.type) {
                case "status":
                    statusEl?.remove();
                    statusEl = addStatusLine(ev.message || "Thinking…");
                    break;
                case "answer_start":
                    statusEl?.remove();
                    answerBubble = addMessage("bot", "<p></p>");
                    break;
                case "token":
                    answerText += ev.value || "";
                    scheduleRender();
                    break;
                case "clarify":
                    statusEl?.remove();
                    answerBubble?.parentElement?.remove();
                    applyProfile(ev.profile);
                    renderClarify(ev.preamble, ev.questions || []);
                    break;
                case "done":
                    statusEl?.remove();
                    applyProfile(ev.profile);
                    if (answerText.trim()) { logEntry.b = answerText; saveChatLog(); }
                    break;
                case "error":
                    statusEl?.remove();
                    answerBubble?.remove();
                    const errBubble = addMessage("bot", null);
                    errBubble.textContent = `⚠️ ${ev.message || "Something went wrong."}`;
                    errBubble.style.background = "var(--danger-bg)";
                    toastMsg(ev.message || "Something went wrong.", true);
                    break;
            }
        };

        try {
            const res = await fetch(`${BACKEND_URL}/api/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId, question: text }),
                signal: activeController.signal,
            });
            if (!res.ok || !res.body) {
                let detail = `Request failed (${res.status})`;
                try { detail = (await res.json()).detail || detail; } catch (e) {}
                throw new Error(detail);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                let sep;
                while ((sep = buffer.indexOf("\n\n")) >= 0) {
                    const frame = buffer.slice(0, sep);
                    buffer = buffer.slice(sep + 2);
                    for (const line of frame.split("\n")) {
                        if (!line.startsWith("data:")) continue;
                        try { handleEvent(JSON.parse(line.slice(5).trim())); }
                        catch (e) { console.warn("Bad SSE frame", line, e); }
                    }
                }
            }
        } catch (err) {
            if (err.name !== "AbortError") {
                statusEl?.remove();
                answerBubble?.remove();
                const errBubble = addMessage("bot", null);
                errBubble.textContent = `⚠️ ${err.message}`;
                errBubble.style.background = "var(--danger-bg)";
                toastMsg(err.message, true);
                logEntry.b = `⚠️ ${err.message}`;
                logEntry.err = true;
                saveChatLog();
            }
        } finally {
            if (answerBubble) {
                answerBubble.classList.remove("streaming");
                if (answerText.trim()) answerBubble.innerHTML = markdown(answerText);
            }
            setBusy(false);
            activeController = null;
            userInput.focus();
        }
    }

    // ------------------------------------------------------------------
    // Uploads
    // ------------------------------------------------------------------
    async function uploadFile(file) {
        if (uploading || busy || !file) return;
        setUploading(true);
        addMessage("user", null).textContent = `📎 Uploading report: ${file.name}`;
        const logEntry = { u: `📎 Uploading report: ${file.name}`, b: null };
        chatLog.push(logEntry);
        saveChatLog();
        const statusEl = addStatusLine(`Reading ${file.name} — extracting, indexing and summarizing…`);

        const formData = new FormData();
        formData.append("session_id", sessionId);
        formData.append("file", file);

        try {
            const res = await fetch(`${BACKEND_URL}/api/upload`, { method: "POST", body: formData });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Upload failed.");
            statusEl.remove();
            const parts = [`**${data.message}**`];
            if (data.summary) parts.push(data.summary);
            addMessage("bot", markdown(parts.join("\n\n")));
            logEntry.b = parts.join("\n\n");
            saveChatLog();
            if (Array.isArray(data.files)) uploadedFiles = data.files;
            applyProfile(data.profile);
            toastMsg(`${file.name} added to your knowledge base.`);
        } catch (err) {
            statusEl.remove();
            const errBubble = addMessage("bot", null);
            errBubble.textContent = `⚠️ ${err.message}`;
            errBubble.style.background = "var(--danger-bg)";
            toastMsg(err.message, true);
            logEntry.b = `⚠️ ${err.message}`;
            logEntry.err = true;
            saveChatLog();
        } finally {
            fileInput.value = "";
            setUploading(false);
        }
    }

    attachButton.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => fileInput.files[0] && uploadFile(fileInput.files[0]));

    ["dragenter", "dragover"].forEach((evt) =>
        chatWindow.addEventListener(evt, (e) => { e.preventDefault(); dragOverlay.hidden = false; }));
    ["dragleave", "drop"].forEach((evt) =>
        chatWindow.addEventListener(evt, (e) => { e.preventDefault(); if (evt === "drop" || e.target === chatWindow) dragOverlay.hidden = true; }));
    chatWindow.addEventListener("drop", (e) => {
        const file = e.dataTransfer?.files?.[0];
        if (file) uploadFile(file);
    });

    // ------------------------------------------------------------------
    // Composer & misc
    // ------------------------------------------------------------------
    composer.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || busy || uploading) return;
        userInput.value = "";
        autoResize();
        sendUserMessage(text);
    });

    const autoResize = () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 140) + "px";
        sendButton.disabled = busy || uploading || !userInput.value.trim();
    };
    userInput.addEventListener("input", autoResize);
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            composer.requestSubmit();
        }
    });

    $("starter-chips")?.addEventListener("click", (e) => {
        const chip = e.target.closest(".chip");
        if (chip) sendUserMessage(chip.dataset.prompt);
    });

    newChatButton.addEventListener("click", async () => {
        if (!confirm("Start a new chat? This clears the conversation and all uploaded reports.")) return;
        activeController?.abort();
        try {
            await fetch(`${BACKEND_URL}/api/session/clear`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: sessionId }),
            });
        } catch (err) { console.warn("Clear failed", err); }
        sessionStorage.removeItem("asha-session");
        sessionStorage.removeItem("asha-chat");
        location.reload();
    });

    restoreChat();
    setBusy(false);
    renderProfileDot();
    userInput.focus();
});
