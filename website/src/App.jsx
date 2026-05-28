import { useMemo, useRef, useState } from 'react'

const scripts = [
  {
    id: 'q1',
    title: 'Q1 Game Search',
    file: 'q1_game_search.py',
    summary: 'Minimax, alpha-beta, heuristic search, and MCTS on Tic-Tac-Toe.',
  },
  {
    id: 'q2',
    title: 'Q2 Travel Planner',
    file: 'q2_travel_planner.py',
    summary: 'Knowledge-base rules for destination scoring and tour recommendations.',
  },
  {
    id: 'q3',
    title: 'Q3 Bayesian Networks',
    file: 'q3_bayesian_networks.py',
    summary: 'Asia network construction, exact inference, and MAP explanation.',
  },
  {
    id: 'q4',
    title: 'Q4 Knowledge Base',
    file: 'q4_knowledge_base.py',
    summary: 'Knowledge graph construction, connectivity analysis, and graph exports.',
  },
]

const statusStyles = {
  idle: 'border-slate-200 bg-white text-slate-600',
  running: 'border-cyan-200 bg-cyan-50 text-cyan-700',
  complete: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  error: 'border-rose-200 bg-rose-50 text-rose-700',
}

function App() {
  const [selectedId, setSelectedId] = useState('q1')
  const [steps, setSteps] = useState([])
  const [activeRun, setActiveRun] = useState(null)
  const [runState, setRunState] = useState('idle')
  const abortRef = useRef(null)

  const selectedScript = useMemo(
    () => scripts.find((script) => script.id === selectedId) ?? scripts[0],
    [selectedId],
  )

  async function runScript(scriptId = selectedId) {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setSelectedId(scriptId)
    setSteps([])
    setActiveRun(scriptId)
    setRunState('running')

    try {
      const response = await fetch(`/api/run/${scriptId}`, {
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        throw new Error(`API returned ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue
          const step = JSON.parse(trimmed)
          setSteps((current) => [
            ...current,
            { ...step, id: crypto.randomUUID(), receivedAt: new Date().toLocaleTimeString() },
          ])
        }
      }

      if (buffer.trim()) {
        const step = JSON.parse(buffer.trim())
        setSteps((current) => [
          ...current,
          { ...step, id: crypto.randomUUID(), receivedAt: new Date().toLocaleTimeString() },
        ])
      }

      setRunState('complete')
    } catch (error) {
      if (error.name === 'AbortError') return
      setRunState('error')
      setSteps((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          title: 'Frontend API error',
          status: 'error',
          detail: error.message,
          data: {},
          receivedAt: new Date().toLocaleTimeString(),
        },
      ])
    } finally {
      setActiveRun(null)
    }
  }

  function stopRun() {
    abortRef.current?.abort()
    setActiveRun(null)
    setRunState('idle')
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-slate-200 pb-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wide text-cyan-700">
              Assignment 5 runner
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-normal text-slate-950 sm:text-4xl">
              Script control panel
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
              Run Q1, Q2, Q3, or Q4 on demand and watch each Python step stream back through the local Vite API.
            </p>
          </div>

          <div className={`w-fit rounded-md border px-3 py-2 text-sm font-medium ${statusStyles[runState]}`}>
            {runState === 'running' ? `Running ${activeRun?.toUpperCase()}` : runState}
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[360px_1fr]">
          <aside className="flex flex-col gap-3">
            {scripts.map((script) => {
              const isSelected = selectedId === script.id
              const isRunning = activeRun === script.id

              return (
                <button
                  key={script.id}
                  type="button"
                  onClick={() => setSelectedId(script.id)}
                  className={`rounded-lg border p-4 text-left transition ${
                    isSelected
                      ? 'border-cyan-500 bg-white shadow-sm'
                      : 'border-slate-200 bg-white hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-base font-semibold text-slate-950">{script.title}</span>
                    <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-600">
                      {script.id.toUpperCase()}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-5 text-slate-600">{script.summary}</p>
                  <p className="mt-3 font-mono text-xs text-slate-500">{script.file}</p>
                  {isRunning ? (
                    <span className="mt-3 inline-flex rounded-md border border-cyan-200 bg-cyan-50 px-2 py-1 text-xs font-medium text-cyan-700">
                      streaming
                    </span>
                  ) : null}
                </button>
              )
            })}

            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => runScript()}
                disabled={runState === 'running'}
                className="flex-1 rounded-md bg-cyan-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                Run selected
              </button>
              <button
                type="button"
                onClick={stopRun}
                disabled={runState !== 'running'}
                className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
              >
                Stop
              </button>
            </div>
          </aside>

          <section className="rounded-lg border border-slate-200 bg-white">
            <div className="flex flex-col gap-3 border-b border-slate-200 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-xl font-semibold text-slate-950">{selectedScript.title}</h2>
                <p className="mt-1 text-sm text-slate-600">{selectedScript.summary}</p>
              </div>
              <button
                type="button"
                onClick={() => runScript(selectedScript.id)}
                disabled={runState === 'running'}
                className="rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                Run {selectedScript.id.toUpperCase()}
              </button>
            </div>

            <div className="max-h-[68vh] min-h-[480px] overflow-auto p-4">
              {steps.length === 0 ? (
                <div className="flex min-h-[420px] items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-center">
                  <p className="max-w-md text-sm leading-6 text-slate-600">
                    Select a question and run it to stream step-by-step output from the Python script.
                  </p>
                </div>
              ) : (
                <ol className="space-y-3">
                  {steps.map((step, index) => (
                    <li key={step.id} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <p className="text-sm font-semibold text-slate-500">Step {index + 1}</p>
                          <h3 className="mt-1 text-lg font-semibold text-slate-950">{step.title}</h3>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className={`rounded-md border px-2 py-1 text-xs font-medium ${statusStyles[step.status] ?? statusStyles.running}`}>
                            {step.status ?? 'running'}
                          </span>
                          <span className="font-mono text-xs text-slate-500">{step.receivedAt}</span>
                        </div>
                      </div>
                      {step.detail ? (
                        <p className="mt-2 text-sm leading-6 text-slate-600">{step.detail}</p>
                      ) : null}
                      {step.data && Object.keys(step.data).length > 0 ? (
                        <pre className="mt-3 overflow-auto rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-100">
                          {JSON.stringify(step.data, null, 2)}
                        </pre>
                      ) : null}
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </section>
        </section>
      </div>
    </main>
  )
}

export default App
