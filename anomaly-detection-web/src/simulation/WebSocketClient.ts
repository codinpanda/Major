export class WebSocketClient {
    private ws: WebSocket | null = null;
    private url: string;
    private onMessageCallback: (data: any) => void;
    private reconnectInterval: number = 3000;

    constructor(url: string, onMessage: (data: any) => void) {
        this.url = url; // e.g., 'ws://localhost:8080'
        this.onMessageCallback = onMessage;
    }

    public connect() {
        if (this.ws) {
            this.ws.close();
        }

        console.log(`Connecting to ${this.url}...`);
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('WebSocket Connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.onMessageCallback(data);
            } catch (e) {
                console.warn('Failed to parse WebSocket message:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket Disconnected');
            // Auto-reconnect
            setTimeout(() => this.connect(), this.reconnectInterval);
        };

        this.ws.onerror = (err) => {
            console.error('WebSocket Error:', err);
            this.ws?.close();
        };
    }

    public disconnect() {
        this.ws?.close();
        this.ws = null;
    }
}
