import React, { useState } from "react"

type BayesianResult = {
  disease: string
  probability: number
}

const App: React.FC = () => {
  const [symptoms, setSymptoms] = useState<string>("")
  const [bayesianOutput, setBayesianOutput] = useState<BayesianResult[]>([])
  const [mragResponse, setMragResponse] = useState<string>("")
  const [loading, setLoading] = useState<boolean>(false)

  // Example dynamic Bayesian predictions
  const runBayesian = () => {
    const lower = symptoms.toLowerCase()
    const output: BayesianResult[] = []

    if (lower.includes("fever") && lower.includes("rash")) {
      output.push({ disease: "Measles", probability: 0.75 })
    } else if (lower.includes("fatigue") && lower.includes("joint pain")) {
      output.push({ disease: "Rheumatoid Arthritis", probability: 0.7 })
    } else if (lower.includes("cough") && lower.includes("fever")) {
      output.push({ disease: "Pneumonia", probability: 0.8 })
    } else {
      output.push({ disease: "Unknown Disease", probability: 0.5 })
    }

    setBayesianOutput(output)
  }

  const runMRAG = async () => {
    if (bayesianOutput.length === 0) return

    setLoading(true)

    // Convert array to dictionary for backend
    const bayesianDict: { [key: string]: number } = {}
    bayesianOutput.forEach(d => {
      bayesianDict[d.disease] = d.probability
    })

    const res = await fetch("http://localhost:8000/mrag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        symptoms: symptoms.split(",").map(s => s.trim()),
        bayesian_output: bayesianDict
      })
    })

    const data = await res.json()
    setMragResponse(data.explanation)
    setLoading(false)
  }

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h1>Medical MRAG Chatbot</h1>

      <textarea
        placeholder="Enter symptoms separated by commas (e.g. fever, rash, headache)"
        value={symptoms}
        onChange={e => setSymptoms(e.target.value)}
        rows={4}
        style={{ width: "100%", marginBottom: "10px" }}
      />

      <button onClick={runBayesian}>Run Bayesian Model</button>

      {bayesianOutput.length > 0 && (
        <>
          <h3>Bayesian Predictions</h3>
          <ul>
            {bayesianOutput.map((item, idx) => (
              <li key={idx}>
                {item.disease} — {(item.probability * 100).toFixed(1)}%
              </li>
            ))}
          </ul>

          <button onClick={runMRAG} disabled={loading}>
            {loading ? "Generating explanation..." : "Explain (MRAG)"}
          </button>
        </>
      )}

      {mragResponse && (
        <>
          <h2>Explanation</h2>
          <pre style={{ whiteSpace: "pre-wrap" }}>{mragResponse}</pre>
        </>
      )}
    </div>
  )
}

export default App
