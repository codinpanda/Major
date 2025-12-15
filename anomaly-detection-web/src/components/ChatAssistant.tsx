import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MessageCircle, X, Send, Bot } from 'lucide-react';

interface Message {
    id: string;
    text: string;
    sender: 'user' | 'bot';
    timestamp: Date;
}

export function ChatAssistant() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([
        { id: '1', text: "Hi! I'm your health assistant. How can I help you today?", sender: 'bot', timestamp: new Date() }
    ]);
    const [input, setInput] = useState("");
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isOpen]);

    const navigate = useNavigate();

    const HEALTH_KNOWLEDGE_BASE = {
        educational: {
            keywords: ['hrv', 'variability', 'rmssd'],
            response: "HRV (Heart Rate Variability) measures the variation in time between heartbeats. Higher HRV generally indicates better fitness and stress resilience."
        },
        symptoms: {
            keywords: ['dizzy', 'headache', 'faint', 'lightheaded'],
            response: "If you feel dizzy or faint, please sit or lie down immediately. Ensure you have fresh air. If symptoms persist for >2 mins, contact a medic."
        },
        panic: {
            keywords: ['panic', 'anxiety', 'anxious', 'scared', 'fear'],
            response: "It sounds like you might be experiencing anxiety. Try the 4-4-4 Box Breathing technique: Inhale for 4s, Hold for 4s, Exhale for 4s. Shall I take you to the Guidance page?",
            action: '/guidance'
        },
        emergency: {
            keywords: ['pain', 'chest', 'stroke', 'emergency', 'dying', 'collapse'],
            response: "CRITICAL ALERT: If you have severe chest pain or difficulty breathing, call Emergency Services immediately. Do not rely on this app for critical life support."
        },
        general_health: {
            keywords: ['hydrate', 'water', 'thirsty'],
            response: "Hydration is key. Vitals can fluctuate if you are dehydrated. enhanced stress readings often correlate with low water intake."
        }
    };

    const processMessage = (text: string) => {
        const lower = text.toLowerCase();

        // 1. Navigation Logic (Priority)
        if (lower.includes('setting') || lower.includes('configure') || lower.includes('profile')) {
            navigate('/settings');
            return "Navigating to Settings. You can configure your alerts and profile there.";
        }
        if (lower.includes('history') || lower.includes('log') || lower.includes('past')) {
            navigate('/history');
            return "Opening your Event Log history.";
        }
        if (lower.includes('guide') || lower.includes('breathing') || lower.includes('help')) {
            navigate('/guidance');
            return "Taking you to the Guidance Protocols page.";
        }
        if (lower.includes('home') || lower.includes('dashboard') || lower.includes('live')) {
            navigate('/');
            return "Returning to the Live Monitor dashboard.";
        }

        // 2. Health Knowledge Base Matching
        // 2. Health Knowledge Base Matching
        for (const data of Object.values(HEALTH_KNOWLEDGE_BASE)) {
            if (data.keywords.some(k => lower.includes(k))) {
                if ('action' in data && data.action) {
                    // Optional navigation can be added here
                }
                return data.response;
            }
        }

        // 3. Fallback / Contextual Vitals
        if (lower.includes('heart') || lower.includes('hr')) return "Your heart rate is currently stable. If it exceeds 110 BPM while resting, I'll alert you.";
        if (lower.includes('stress')) return "I'm monitoring your HRV. Try the Box Breathing exercise if you feel overwhelmed.";
        if (lower.includes('anomaly') || lower.includes('alert')) return "I detected a minor anomaly 5 minutes ago, but your vitals have normalized.";
        if (lower.includes('hello') || lower.includes('hi')) return "Hello! I can help with navigation ('Go to Settings') or health questions ('I feel dizzy').";

        return "I can help with health insights or navigation. Try asking about 'HRV', 'Anxiety', or say 'Go home'.";
    };

    const handleSend = () => {
        if (!input.trim()) return;

        const userMsg: Message = {
            id: Date.now().toString(),
            text: input,
            sender: 'user',
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMsg]);
        setInput("");

        // Simulated Bot Response
        setTimeout(() => {
            const botResponse = processMessage(input);
            const botMsg: Message = {
                id: (Date.now() + 1).toString(),
                text: botResponse,
                sender: 'bot',
                timestamp: new Date()
            };
            setMessages(prev => [...prev, botMsg]);
        }, 800);
    };

    return (
        <>
            {/* Floating Toggle Button - Touch-friendly */}
            <button
                onClick={() => setIsOpen(true)}
                className={`fixed bottom-20 right-4 sm:bottom-6 sm:right-6 p-3 sm:p-4 rounded-full bg-blue-600 text-white shadow-lg hover:bg-blue-700 transition-all z-40 tap-target ${isOpen ? 'scale-0' : 'scale-100'}`}
            >
                <MessageCircle size={20} className="sm:w-6 sm:h-6" />
            </button>

            {/* Chat Window - Full screen on mobile, floating on desktop */}
            <div className={`fixed inset-0 sm:inset-auto sm:bottom-6 sm:right-6 sm:w-96 sm:h-[500px] bg-[#161b22] border-0 sm:border border-[#24292e] sm:rounded-2xl shadow-2xl flex flex-col transition-all z-50 ${isOpen ? 'scale-100 opacity-100' : 'scale-0 opacity-0 pointer-events-none'}`}>

                {/* Header - Responsive */}
                <div className="p-3 sm:p-4 border-b border-[#24292e] flex justify-between items-center bg-[#0d1117] sm:rounded-t-2xl">
                    <div className="flex items-center gap-2">
                        <div className="p-1.5 bg-blue-600/20 text-blue-400 rounded-lg">
                            <Bot size={18} className="sm:w-5 sm:h-5" />
                        </div>
                        <span className="text-sm sm:text-base font-semibold text-gray-200">Assistant</span>
                    </div>
                    <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white tap-target">
                        <X size={20} className="sm:w-5 sm:h-5" />
                    </button>
                </div>

                {/* Messages Area - Responsive */}
                <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-3 sm:space-y-4 smooth-scroll">
                    {messages.map(msg => (
                        <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[85%] sm:max-w-[80%] p-2.5 sm:p-3 rounded-2xl text-xs sm:text-sm ${msg.sender === 'user'
                                ? 'bg-blue-600 text-white rounded-br-none'
                                : 'bg-[#24292e] text-gray-200 rounded-bl-none'
                                }`}>
                                {msg.text}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area - Mobile-optimized */}
                <div className="p-3 sm:p-4 border-t border-[#24292e] flex gap-2 bg-[#0d1117]">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="Ask about your health..."
                        className="flex-1 bg-[#161b22] border border-[#30363d] rounded-xl px-3 py-2.5 sm:py-2 text-xs sm:text-sm text-gray-200 focus:outline-none focus:border-blue-500"
                    />
                    <button
                        onClick={handleSend}
                        className="p-2.5 sm:p-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors tap-target"
                    >
                        <Send size={16} className="sm:w-[18px] sm:h-[18px]" />
                    </button>
                </div>
            </div>
        </>
    );
}
