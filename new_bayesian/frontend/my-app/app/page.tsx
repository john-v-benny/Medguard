'use client';

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, Activity, RefreshCw, Check, Stethoscope, Thermometer, User } from 'lucide-react';
import clsx from 'clsx';

// --- Types ---
type Message = { role: 'user' | 'assistant'; content: string; };
type Symptoms = { [key: string]: any; };
type PredictionResult = {
  prediction: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  heatmap_b64: string;
};

const FEATURES = ["Fever", "Headache", "Cough", "Fatigue", "Body_Pain"];
const API_URL = "http://localhost:8000/api";

export default function MedicalAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: "Hello. I'm your AI medical assistant. Let's check your vitals. First, could you tell me if you have a fever?" }
  ]);
  const [input, setInput] = useState('');
  const [collected, setCollected] = useState<Symptoms>({
    Fever: null, Headache: null, Cough: null, Fatigue: null, Body_Pain: null
  });
  
  // Generic state to hold the value of the active widget
  const [widgetValue, setWidgetValue] = useState<number | string>('');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // --- Helpers ---
  const getNextMissingSymptom = (currentCollected: Symptoms) => {
    return FEATURES.find(f => currentCollected[f] === null || currentCollected[f] === undefined) || null;
  };

  const currentPendingSymptom = getNextMissingSymptom(collected);

  // --- Effect: Set specific start values ---
  useEffect(() => {
    if (currentPendingSymptom === 'Fever') {
        setWidgetValue(98.5); 
    } else {
        setWidgetValue(0); 
    }
  }, [currentPendingSymptom]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentPendingSymptom, result]);

  // --- Logic ---
  const handleSendMessage = async (text: string, overrideCollected?: Symptoms) => {
    if (!text.trim()) return;

    const currentCollected = overrideCollected || collected;
    const newHistory = [...messages, { role: 'user', content: text } as Message];
    
    setMessages(newHistory);
    setInput('');
    setLoading(true);

    try {
      const { data } = await axios.post(`${API_URL}/chat`, {
        history: newHistory,
        collected: currentCollected
      });

      const newCollected = { ...currentCollected };
      if (data.updates) {
        Object.keys(data.updates).forEach(key => {
            if (data.updates[key] !== null) {
                newCollected[key] = data.updates[key];
            }
        });
      }
      setCollected(newCollected);

      if (data.acknowledgment) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.acknowledgment }]);
      }

      const nextMissing = getNextMissingSymptom(newCollected);
      if (!nextMissing) {
        await fetchPrediction(newCollected);
      }

    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'assistant', content: "I'm having trouble connecting. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const fetchPrediction = async (finalSymptoms: Symptoms) => {
    setLoading(true);
    try {
      const cleanSymptoms: Record<string, number> = {};
      FEATURES.forEach(f => {
        const val = finalSymptoms[f];
        cleanSymptoms[f] = typeof val === 'object' && val !== null ? parseFloat(val.value || val.severity || 0) : parseFloat(val);
      });

      const { data } = await axios.post(`${API_URL}/predict`, { symptoms: cleanSymptoms });
      setResult(data);
    } catch (error) {
      console.error("Prediction Error", error);
    } finally {
      setLoading(false);
    }
  };

  const submitWidgetValue = (val: number | string) => {
    if (!currentPendingSymptom) return;
    
    const numVal = parseFloat(val.toString());
    if (isNaN(numVal)) return;

    const updatedCollected = { ...collected, [currentPendingSymptom]: numVal };
    setCollected(updatedCollected);
    
    const unit = currentPendingSymptom === 'Fever' ? '°F' : '/10';
    const userMsg = `My ${currentPendingSymptom} is ${numVal}${unit}`;
    
    handleSendMessage(userMsg, updatedCollected);
  };

  const restart = () => {
    setCollected({ Fever: null, Headache: null, Cough: null, Fatigue: null, Body_Pain: null });
    setMessages([{ role: 'assistant', content: "Hello. I'm your AI medical assistant. Let's check your vitals. First, could you tell me if you have a fever?" }]);
    setResult(null);
    setWidgetValue(98.5); 
  };

  const getDisplayValue = (val: any) => {
    if (val === null || val === undefined) return null;
    if (typeof val === 'object') return val.value || val.severity || 0;
    return val;
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-800 overflow-hidden">
      
      {/* Sidebar - Desktop Only */}
      <aside className="hidden md:flex w-80 bg-white border-r border-slate-200 flex-col p-6 shadow-sm z-10">
        <div className="flex items-center gap-3 mb-8">
          <div className="bg-indigo-600 p-2.5 rounded-xl shadow-lg shadow-indigo-200">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900 leading-none">MedGuard</h1>
            <span className="text-xs text-slate-500 font-medium">AI Diagnostic Tool</span>
          </div>
        </div>
        
        {/* Vertical Stepper */}
        <div className="flex-1 overflow-y-auto space-y-1 pr-2">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">Vitals Check</h3>
            {FEATURES.map((f, i) => {
              const val = getDisplayValue(collected[f]);
              const isDone = val !== null;
              const isCurrent = f === currentPendingSymptom && !result;
              
              return (
                <div key={f} 
                  className={clsx(
                    "flex items-center p-3 rounded-lg transition-all duration-200",
                    isCurrent ? "bg-indigo-50 border border-indigo-100 shadow-sm" : "hover:bg-slate-50",
                    isDone ? "opacity-100" : "opacity-60"
                  )}
                >
                  <div className={clsx(
                    "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold mr-3 border transition-colors",
                    isDone ? "bg-green-100 text-green-700 border-green-200" : 
                    isCurrent ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-slate-400 border-slate-200"
                  )}>
                    {isDone ? <Check className="w-4 h-4" /> : i + 1}
                  </div>
                  <div className="flex-1">
                    <div className={clsx("text-sm font-medium", isCurrent ? "text-indigo-900" : "text-slate-700")}>{f}</div>
                    <div className="text-xs text-slate-500">{isDone ? `Recorded: ${val}` : isCurrent ? 'Waiting...' : 'Pending'}</div>
                  </div>
                </div>
              );
            })}
        </div>
        
        <button onClick={restart} className="mt-6 flex items-center justify-center w-full gap-2 px-4 py-3 text-sm font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-xl transition-colors">
          <RefreshCw className="w-4 h-4" /> Reset Session
        </button>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col relative w-full">
        
        {/* Mobile Header */}
        <div className="md:hidden h-16 bg-white border-b border-slate-200 flex items-center px-4 justify-between shrink-0">
           <div className="flex items-center gap-2">
             <div className="bg-indigo-600 p-1.5 rounded-lg"><Activity className="w-5 h-5 text-white" /></div>
             <span className="font-bold text-slate-800">MedGuard</span>
           </div>
           <button onClick={restart} className="p-2 text-slate-500"><RefreshCw className="w-5 h-5"/></button>
        </div>

        {/* Messages Scroll Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scroll-smooth">
          {messages.map((msg, idx) => (
            <div key={idx} className={clsx("flex w-full", msg.role === 'user' ? 'justify-end' : 'justify-start')}>
              <div className={clsx(
                "flex max-w-[85%] md:max-w-[70%] rounded-2xl p-4 shadow-sm",
                msg.role === 'user' ? "bg-indigo-600 text-white rounded-br-none" : "bg-white border border-slate-100 text-slate-800 rounded-bl-none"
              )}>
                <div className="mr-3 mt-1 shrink-0 opacity-80">
                  {msg.role === 'assistant' ? <Stethoscope className="w-5 h-5" /> : <User className="w-5 h-5" />}
                </div>
                <div className="leading-relaxed text-sm md:text-base">{msg.content}</div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-slate-100 px-4 py-3 rounded-2xl rounded-bl-none shadow-sm flex items-center gap-1">
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-75" />
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-150" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} className="h-4" />
        </div>

        {/* Bottom Interaction Area */}
        <div className="bg-white/80 backdrop-blur-md border-t border-slate-200 p-4 md:p-6 shrink-0">
          <div className="max-w-4xl mx-auto w-full space-y-4">
            
            {/* 1. Results Panel */}
            {result && (
              <div className="bg-emerald-50 border border-emerald-100 rounded-2xl p-4 md:p-6 animate-fade-in shadow-sm">
                 {/* --- UPDATED LAYOUT FOR LARGER IMAGE --- */}
                 <div className="flex flex-col md:flex-row gap-8">
                    <div className="flex-1 space-y-4">
                        <div className="flex items-center gap-2 text-emerald-800 font-bold text-lg">
                           <Activity className="w-6 h-6" /> Diagnosis Complete
                        </div>
                        <div>
                            <div className="text-sm text-emerald-600 uppercase tracking-wide font-semibold mb-1">Likely Condition</div>
                            <div className="text-3xl font-bold text-slate-900">{result.prediction}</div>
                            <div className="text-emerald-700 font-medium mt-1">{(result.confidence * 100).toFixed(1)}% Match Confidence</div>
                        </div>
                        <div className="space-y-2 mt-4">
                             {Object.entries(result.all_probabilities).sort(([,a], [,b]) => b - a).slice(0, 3).map(([d, p]) => (
                                 <div key={d} className="flex items-center text-xs md:text-sm">
                                     <span className="w-24 truncate font-medium text-slate-600">{d}</span>
                                     <div className="flex-1 h-2 bg-white rounded-full overflow-hidden mx-2 border border-emerald-100">
                                         <div className="h-full bg-emerald-500 rounded-full" style={{width: `${p*100}%`}}/>
                                     </div>
                                     <span className="text-slate-500 w-10 text-right">{(p*100).toFixed(0)}%</span>
                                 </div>
                             ))}
                        </div>
                    </div>
                    {/* --- IMAGE CONTAINER INCREASED TO 50% --- */}
                    {result.heatmap_b64 && (
                        <div className="md:w-1/2 bg-white p-2 rounded-xl border border-emerald-100 shadow-sm flex items-center justify-center">
                            <img src={`data:image/png;base64,${result.heatmap_b64}`} className="w-full h-auto max-h-[500px] object-contain rounded-lg" alt="Analysis" />
                        </div>
                    )}
                 </div>
              </div>
            )}

            {/* 2. Interactive Widget */}
            {!result && currentPendingSymptom && (
               <div className="bg-indigo-50/50 border border-indigo-100 rounded-xl p-4 flex flex-col md:flex-row items-center gap-4 animate-in slide-in-from-bottom-4 fade-in">
                  <div className="flex items-center gap-3 w-full md:w-auto">
                     <div className="bg-white p-2 rounded-full shadow-sm text-indigo-600">
                        {currentPendingSymptom === 'Fever' ? <Thermometer className="w-5 h-5"/> : <Activity className="w-5 h-5"/>}
                     </div>
                     <div className="font-semibold text-indigo-900 text-sm whitespace-nowrap">
                        {currentPendingSymptom === 'Fever' ? 'Enter Temperature' : `Rate ${currentPendingSymptom}`}
                     </div>
                  </div>
                  
                  <div className="flex-1 w-full">
                      {currentPendingSymptom === 'Fever' ? (
                          <div className="flex gap-2">
                             <input 
                                type="number" 
                                placeholder="98.6" 
                                min="0"
                                value={widgetValue}
                                onChange={(e) => setWidgetValue(e.target.value)}
                                className="w-full px-4 py-2 rounded-lg border-indigo-200 focus:ring-2 focus:ring-indigo-500 outline-none" 
                                onKeyDown={(e) => e.key === 'Enter' && submitWidgetValue(widgetValue)}
                             />
                             <button 
                                onClick={() => submitWidgetValue(widgetValue)} 
                                className="bg-indigo-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-indigo-700"
                             >
                                Submit
                             </button>
                          </div>
                      ) : (
                          <div className="w-full px-2">
                             <div className="flex justify-center mb-2">
                                <span className="bg-indigo-100 text-indigo-800 font-bold px-3 py-1 rounded-md text-sm">
                                    Severity: {widgetValue} / 10
                                </span>
                             </div>

                             <input 
                                type="range" 
                                min="0" 
                                max="10" 
                                value={typeof widgetValue === 'number' ? widgetValue : 0} 
                                onChange={(e) => setWidgetValue(parseFloat(e.target.value))}
                                className="w-full h-2 bg-indigo-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" 
                             />
                             
                             <div className="flex justify-between text-[10px] text-indigo-400 font-bold uppercase mt-1 px-1">
                                <span>None</span><span>Moderate</span><span>Severe</span>
                             </div>

                             <button 
                                onClick={() => submitWidgetValue(widgetValue)}
                                className="mt-3 w-full bg-indigo-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-indigo-700 shadow-sm transition-colors"
                             >
                                Submit Value
                             </button>
                          </div>
                      )}
                  </div>
               </div>
            )}

            {/* 3. Text Input */}
            {!result && (
                <div className="relative">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage(input)}
                    placeholder={currentPendingSymptom ? `Or describe your ${currentPendingSymptom.toLowerCase()}...` : "Type a message..."}
                    className="w-full pl-5 pr-14 py-4 bg-slate-100 hover:bg-white focus:bg-white border-0 focus:ring-2 ring-indigo-500/20 rounded-2xl transition-all shadow-inner text-slate-800 placeholder:text-slate-400"
                    disabled={loading}
                />
                <button 
                    onClick={() => handleSendMessage(input)}
                    disabled={loading || !input.trim()}
                    className="absolute right-2 top-1/2 -translate-y-1/2 bg-indigo-600 hover:bg-indigo-700 text-white p-2.5 rounded-xl disabled:opacity-50 disabled:hover:bg-indigo-600 transition-colors shadow-sm"
                >
                    <Send className="w-4 h-4" />
                </button>
                </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}