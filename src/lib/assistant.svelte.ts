import {
  validateAssistantRequest, validateProposal,
  type AssistantCommand, type AssistantContext, type AssistantMessage, type AssistantProposal,
} from "../assistant/contract.ts";

export type ProposalStatus = "pending" | "applying" | "applied" | "dismissed" | "superseded" | "failed";
export interface ChatEntry {
  id: string;
  role: "user" | "assistant";
  content: string;
  proposal?: AssistantProposal;
  context?: AssistantContext;
  proposalStatus?: ProposalStatus;
  applicationResult?: string;
  applicationError?: string;
}

const MAX_PROMPT = 8000;
const MAX_HISTORY = 12;
const MAX_HISTORY_CHARS = 24000;

export class AssistantSession {
  consent = $state(false);
  draft = $state("");
  messages = $state<ChatEntry[]>([]);
  configured = $state(false);
  model = $state<string | null>(null);
  statusLoading = $state(false);
  statusChecked = $state(false);
  connectionError = $state<string | null>(null);
  busy = $state(false);
  applying = $state(false);
  error = $state<string | null>(null);
  notice = $state<string | null>(null);
  retryAvailable = $state(false);
  #request: AbortController | null = null;
  #statusRequest: AbortController | null = null;
  #generation = 0;
  #lastUserId: string | null = null;

  get canSend(): boolean {
    return this.consent && this.configured && !this.busy && !this.applying
      && this.draft.trim().length > 0 && this.draft.trim().length <= MAX_PROMPT;
  }

  async refreshStatus(): Promise<void> {
    this.#statusRequest?.abort();
    const controller = new AbortController();
    this.#statusRequest = controller;
    const timeout = setTimeout(() => controller.abort(), 15000);
    this.statusLoading = true;
    this.connectionError = null;
    try {
      const response = await fetch("/api/assistant/status", {
        cache: "no-store", credentials: "same-origin", signal: controller.signal,
      });
      const data = await this.#read(response) as { configured?: unknown; model?: unknown };
      if (typeof data.configured !== "boolean") throw new Error("Invalid assistant connection status.");
      if (this.#statusRequest !== controller) return;
      this.configured = data.configured;
      this.model = typeof data.model === "string" ? data.model.slice(0, 100) : null;
    } catch (error) {
      if (this.#statusRequest !== controller) return;
      this.configured = false;
      this.connectionError = controller.signal.aborted ? "Connection check timed out." : this.#message(error);
    } finally {
      clearTimeout(timeout);
      if (this.#statusRequest === controller) {
        this.statusLoading = false;
        this.statusChecked = true;
        this.#statusRequest = null;
      }
    }
  }

  #history(): AssistantMessage[] {
    const history: AssistantMessage[] = this.messages.slice(-MAX_HISTORY).map((entry) => {
      const status = entry.proposalStatus ? `\nProposal status: ${entry.proposalStatus}.` : "";
      return { role: entry.role, content: entry.content.slice(0, MAX_PROMPT - status.length) + status };
    });
    while (history.length > 1 && history.reduce((sum, item) => sum + item.content.length, 0) > MAX_HISTORY_CHARS) history.shift();
    return history;
  }

  async send(getContext: () => Promise<AssistantContext>, retry = false): Promise<void> {
    if (this.busy || this.applying) return;
    if (!this.consent) { this.error = "Context sharing must be enabled before sending."; return; }
    if (!this.configured) { this.error = "OpenAI is not configured on this server."; return; }
    if (retry) {
      if (!this.retryAvailable || this.messages.at(-1)?.id !== this.#lastUserId) return;
    } else {
      const content = this.draft.trim();
      if (!content || content.length > MAX_PROMPT) { this.error = "Enter a message of 1 to 8,000 characters."; return; }
      for (const entry of this.messages) if (entry.proposalStatus === "pending") entry.proposalStatus = "superseded";
      const id = crypto.randomUUID();
      this.messages = [...this.messages.slice(-39), { id, role: "user", content }];
      this.#lastUserId = id;
      this.draft = "";
    }
    const generation = ++this.#generation;
    const controller = new AbortController();
    this.#request = controller;
    this.busy = true;
    this.error = null;
    this.notice = null;
    this.retryAvailable = false;
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 120000);
    try {
      // Context loading can stall independently of fetch, so cancellation and
      // the timeout must cover both stages of the request.
      let abortContext = () => {};
      const context = await new Promise<AssistantContext>((resolve, reject) => {
        abortContext = () => reject(new DOMException("Request aborted", "AbortError"));
        controller.signal.addEventListener("abort", abortContext, { once: true });
        Promise.resolve().then(getContext).then(resolve, reject);
      }).finally(() => controller.signal.removeEventListener("abort", abortContext));
      if (generation !== this.#generation || controller.signal.aborted) return;
      const request = validateAssistantRequest({ messages: this.#history(), context });
      const response = await fetch("/api/assistant/chat", {
        method: "POST", credentials: "same-origin", signal: controller.signal,
        headers: { "Content-Type": "application/json", "X-Microduck-Assistant": "1" },
        body: JSON.stringify(request),
      });
      const proposal = validateProposal(await this.#read(response), request.context);
      if (generation !== this.#generation) return;
      const hasChanges = proposal.world !== null || proposal.motion !== null || proposal.training !== null || !!proposal.command;
      this.messages = [...this.messages, {
        id: crypto.randomUUID(), role: "assistant", content: proposal.message,
        ...(hasChanges ? { proposal, context: request.context, proposalStatus: "pending" as const } : {}),
      }];
    } catch (error) {
      if (generation !== this.#generation) return;
      this.error = timedOut ? "The assistant request timed out. You can retry it." : this.#message(error);
      this.retryAvailable = true;
    } finally {
      clearTimeout(timeout);
      if (generation === this.#generation) { this.busy = false; this.#request = null; }
    }
  }

  cancel(): void {
    if (!this.busy) return;
    this.#generation++;
    this.#request?.abort();
    this.#request = null;
    this.busy = false;
    this.notice = "Request canceled.";
    this.error = null;
    this.retryAvailable = true;
  }

  dismiss(id: string): void {
    const entry = this.messages.find((message) => message.id === id);
    if (this.busy || this.applying || entry?.proposalStatus !== "pending") return;
    entry.proposalStatus = "dismissed";
    entry.applicationError = undefined;
  }

  async apply(id: string, applyProposal: (proposal: AssistantProposal, context: AssistantContext) => Promise<string>): Promise<void> {
    const entry = this.messages.find((message) => message.id === id);
    if (this.busy || this.applying || entry?.proposalStatus !== "pending" || !entry.proposal || entry.proposal.command || !entry.context) return;
    this.applying = true;
    entry.proposalStatus = "applying";
    entry.applicationError = undefined;
    try {
      const result = await applyProposal($state.snapshot(entry.proposal), $state.snapshot(entry.context));
      entry.applicationResult = result;
      entry.proposalStatus = "applied";
    } catch (error) {
      entry.proposalStatus = "pending";
      entry.applicationError = this.#message(error);
    } finally { this.applying = false; }
  }

  async execute(id: string, executeCommand: (command: AssistantCommand, context: AssistantContext) => Promise<string>): Promise<void> {
    const entry = this.messages.find((message) => message.id === id);
    if (this.busy || this.applying || entry?.proposalStatus !== "pending" || !entry.proposal?.command || !entry.context) return;
    this.applying = true;
    entry.proposalStatus = "applying";
    entry.applicationError = undefined;
    try {
      entry.applicationResult = await executeCommand($state.snapshot(entry.proposal.command), $state.snapshot(entry.context));
      entry.proposalStatus = "applied";
    } catch (error) {
      // A rejected request may already have begun locally. Never redispatch
      // the same command card; a fresh proposal must capture current state.
      entry.proposalStatus = "failed";
      entry.applicationError = this.#message(error);
    } finally { this.applying = false; }
  }

  async #read(response: Response): Promise<unknown> {
    const raw = await response.text();
    if (raw.length > 262144) throw new Error("Assistant response exceeded the size limit.");
    let data: unknown;
    try { data = JSON.parse(raw); } catch { throw new Error("The assistant server returned an invalid response."); }
    if (!response.ok) {
      const payload = data as { error?: unknown };
      const error = payload?.error;
      const detail = error && typeof error === "object" && !Array.isArray(error)
        ? (error as { message?: unknown }).message : error;
      const message = typeof detail === "string" && detail.trim()
        ? detail : `Assistant request failed (HTTP ${response.status}).`;
      throw new Error(message.slice(0, 500));
    }
    return data;
  }

  #message(error: unknown): string {
    return (error instanceof Error ? error.message : String(error)).slice(0, 500);
  }

  dispose(): void {
    this.#generation++;
    this.#request?.abort();
    this.#statusRequest?.abort();
    this.#request = null;
    this.#statusRequest = null;
  }
}
