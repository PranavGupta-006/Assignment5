import { useMemo, useRef, useState } from 'react'

const scripts = [
  {
    id: 'q1',
    title: 'Game Search',
    label: 'Q1',
    file: 'q1_game_search.py',
    color: 'bg-[#00A7E1]',
    text: 'text-[#0077A3]',
    tint: 'bg-[#E5F8FF]',
    summary: 'Minimax, Alpha-Beta, heuristic Alpha-Beta, and MCTS on Tic-Tac-Toe.',
  },
  {
    id: 'q2',
    title: 'Travel Planner',
    label: 'Q2',
    file: 'q2_travel_planner.py',
    color: 'bg-[#24B47E]',
    text: 'text-[#147A56]',
    tint: 'bg-[#E9FFF5]',
    summary: 'A knowledge-base planner using destinations, food, hotels, costs, and KG tools.',
  },
  {
    id: 'q3',
    title: 'Bayesian Networks',
    label: 'Q3',
    file: 'q3_bayesian_networks.py',
    color: 'bg-[#FF5A5F]',
    text: 'text-[#B83236]',
    tint: 'bg-[#FFF0F1]',
    summary: 'Asia chest-clinic network with CPDs, exact inference, and MAP explanation.',
  },
]

const statusCopy = {
  idle: 'Ready',
  running: 'Running',
  complete: 'Done',
  error: 'Check output',
}

const statusStyle = {
  idle: 'border-slate-200 bg-white text-slate-600',
  running: 'border-[#FFD166] bg-[#FFF7D6] text-[#8A6400]',
  complete: 'border-[#24B47E]/30 bg-[#E9FFF5] text-[#147A56]',
  error: 'border-[#FF5A5F]/30 bg-[#FFF0F1] text-[#B83236]',
}

function App2() {
  const [selectedId, setSelectedId] = useState('q1')
  const [steps, setSteps] = useState([])
  const [activeRun, setActiveRun] = useState(null)
  const [runState, setRunState] = useState('idle')
  const abortRef = useRef(null)

  const selectedScript = useMemo(
    () => scripts.find((script) => script.id === selectedId) ?? scripts[0],
    [selectedId],
  )

  const progress = runState === 'complete' ? 100 : Math.min(94, steps.length * 11)
  const latestStep = steps.at(-1)

  async function runScript(scriptId = selectedId) {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setSelectedId(scriptId)
    setSteps([])
    setActiveRun(scriptId)
    setRunState('running')

    try {
      const response = await fetch(`/api/run/${scriptId}`, { signal: controller.signal })
      if (!response.ok || !response.body) throw new Error(`API returned ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        lines.forEach(addStepLine)
      }

      addStepLine(buffer)
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

  function addStepLine(line) {
    const trimmed = line.trim()
    if (!trimmed) return
    const step = JSON.parse(trimmed)
    setSteps((current) => [
      ...current,
      { ...step, id: crypto.randomUUID(), receivedAt: new Date().toLocaleTimeString() },
    ])
  }

  function stopRun() {
    abortRef.current?.abort()
    setActiveRun(null)
    setRunState('idle')
  }

  return (
    <main className="min-h-screen bg-[#FAFAF7] text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <div className="grid gap-6 lg:grid-cols-[1fr_360px] lg:items-end">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-md bg-[#7C3AED] px-3 py-1 text-xs font-black uppercase text-white">
                  Assignment 5
                </span>
                <span className={`rounded-md border px-3 py-1 text-xs font-black uppercase ${statusStyle[runState]}`}>
                  {statusCopy[runState]}
                </span>
              </div>
              <h1 className="mt-4 text-4xl font-black tracking-normal sm:text-5xl">
                AI Methods, Visualized
              </h1>
              <p className="mt-3 max-w-2xl text-base leading-7 text-slate-600">
                A clean runner for search algorithms, a knowledge-graph travel planner, and Bayesian-network inference.
              </p>
            </div>

            <div className="rounded-lg border border-slate-200 bg-[#111827] p-5 text-white">
              <p className="text-xs font-black uppercase text-[#FFD166]">Submitted by</p>
              <p className="mt-2 text-2xl font-black">Pranav Gupta</p>
              <p className="mt-1 font-mono text-sm text-slate-300">SE24UCSE020</p>
            </div>
          </div>

          <div className="mt-6 grid gap-3 md:grid-cols-3">
            {scripts.map((script) => (
              <ModuleButton
                key={script.id}
                script={script}
                selected={selectedId === script.id}
                busy={runState === 'running'}
                running={activeRun === script.id}
                onSelect={() => setSelectedId(script.id)}
                onRun={() => runScript(script.id)}
              />
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[320px_1fr] lg:px-8">
        <aside className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <p className="text-xs font-black uppercase text-slate-500">Current module</p>
            <h2 className="mt-2 text-2xl font-black">{selectedScript.title}</h2>
            <p className="mt-2 text-sm leading-6 text-slate-600">{selectedScript.summary}</p>
            <p className="mt-4 font-mono text-xs text-slate-500">{selectedScript.file}</p>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
              <div className={`h-full ${selectedScript.color} transition-all duration-300`} style={{ width: `${progress}%` }} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 rounded-lg border border-slate-200 bg-white p-3">
            <button
              type="button"
              onClick={() => runScript()}
              disabled={runState === 'running'}
              className="rounded-md bg-slate-950 px-3 py-2 text-sm font-black text-white transition hover:bg-[#7C3AED] disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              Run
            </button>
            <button
              type="button"
              onClick={stopRun}
              disabled={runState !== 'running'}
              className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-black text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-300"
            >
              Stop
            </button>
          </div>

          {latestStep ? (
            <div className={`rounded-lg border border-slate-200 ${selectedScript.tint} p-4`}>
              <p className={`text-xs font-black uppercase ${selectedScript.text}`}>Latest</p>
              <p className="mt-2 text-sm font-black">{latestStep.title}</p>
              <p className="mt-1 text-sm leading-6 text-slate-700">{latestStep.detail || 'Step completed.'}</p>
            </div>
          ) : null}
        </aside>

        <section className="min-h-[640px] rounded-lg border border-slate-200 bg-white p-4">
          {steps.length === 0 ? (
            <EmptyState script={selectedScript} onRun={() => runScript(selectedScript.id)} />
          ) : (
            <div className="space-y-4">
              <RunSummary steps={steps} script={selectedScript} />
              <ol className="space-y-3">
                {steps.map((step, index) => (
                  <StepCard key={step.id} step={step} index={index} script={selectedScript} />
                ))}
              </ol>
            </div>
          )}
        </section>
      </section>
    </main>
  )
}

function ModuleButton({ script, selected, running, busy, onSelect, onRun }) {
  return (
    <article className={`rounded-lg border bg-white p-4 transition ${selected ? 'border-slate-950 shadow-sm' : 'border-slate-200 hover:border-slate-400'}`}>
      <button type="button" onClick={onSelect} className="block w-full text-left">
        <div className="flex items-center justify-between gap-3">
          <span className={`rounded-md px-3 py-1 text-xs font-black text-white ${script.color}`}>{script.label}</span>
          {running ? <span className="rounded-md bg-[#FFF7D6] px-2 py-1 text-xs font-black text-[#8A6400]">live</span> : null}
        </div>
        <h3 className="mt-3 text-lg font-black">{script.title}</h3>
        <p className="mt-2 text-sm leading-5 text-slate-600">{script.summary}</p>
      </button>
      <button
        type="button"
        onClick={onRun}
        disabled={busy}
        className={`mt-4 w-full rounded-md px-3 py-2 text-sm font-black text-white transition ${script.color} hover:brightness-95 disabled:cursor-not-allowed disabled:bg-slate-300`}
      >
        Run {script.label}
      </button>
    </article>
  )
}

function EmptyState({ script, onRun }) {
  return (
    <div className={`flex min-h-[600px] items-center justify-center rounded-lg ${script.tint} p-6 text-center`}>
      <div className="max-w-md">
        <div className={`mx-auto flex h-16 w-16 items-center justify-center rounded-lg text-xl font-black text-white ${script.color}`}>
          {script.label}
        </div>
        <h3 className="mt-4 text-2xl font-black">Pick a module and run it.</h3>
        <p className="mt-2 text-sm leading-6 text-slate-700">
          The output will appear as compact visual cards instead of a wall of raw terminal text.
        </p>
        <button
          type="button"
          onClick={onRun}
          className={`mt-5 rounded-md px-5 py-2 text-sm font-black text-white ${script.color} hover:brightness-95`}
        >
          Run {script.label}
        </button>
      </div>
    </div>
  )
}

function RunSummary({ steps, script }) {
  const completed = steps.filter((step) => step.status === 'complete').length
  return (
    <section className="grid gap-3 sm:grid-cols-3">
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <p className="text-xs font-black uppercase text-slate-500">Steps streamed</p>
        <p className="mt-1 text-3xl font-black">{steps.length}</p>
      </div>
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
        <p className="text-xs font-black uppercase text-slate-500">Completed</p>
        <p className="mt-1 text-3xl font-black">{completed}</p>
      </div>
      <div className={`rounded-lg border border-slate-200 ${script.tint} p-4`}>
        <p className={`text-xs font-black uppercase ${script.text}`}>Mode</p>
        <p className="mt-1 text-3xl font-black">{script.label}</p>
      </div>
    </section>
  )
}

function StepCard({ step, index, script }) {
  return (
    <li className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex gap-3">
          <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-sm font-black text-white ${script.color}`}>
            {index + 1}
          </span>
          <div>
            <h3 className="text-base font-black">{step.title}</h3>
            {step.detail ? <p className="mt-1 text-sm leading-6 text-slate-600">{step.detail}</p> : null}
          </div>
        </div>
        <span className={`w-fit rounded-md border px-2 py-1 text-xs font-black ${statusStyle[step.status] ?? statusStyle.running}`}>
          {statusCopy[step.status] ?? 'Running'}
        </span>
      </div>
      <VisualData data={step.data} script={script} />
    </li>
  )
}

function VisualData({ data, script }) {
  if (!data || Object.keys(data).length === 0) return null

  return (
    <div className="mt-4 space-y-3">
      <Metrics data={data} script={script} />
      <BoardPanel data={data} script={script} />
      <SearchBars data={data} script={script} />
      <MctsPanel stats={data.child_stats} script={script} />
      <RankedScores scores={data.top_scores} weights={data.score_weights} />
      <ToolCatalog tools={data.tools || data.tooling} />
      <KnowledgeGraph edges={data.graph_edges} />
      <BayesianGraph nodes={data.nodes} edges={data.edges} labels={data.node_labels} />
      <ProbabilityGrid values={data.p_yes} />
      <ComparisonTable rows={data.comparisons} />
      <TourPlan data={data} />
      <StringList title="Attractions" items={data.attractions} />
      <StringList title="Food picks" items={data.cuisines} />
      <StringList title="Notes" items={data.notes} />
    </div>
  )
}

function Metrics({ data, script }) {
  const keys = ['best_move', 'value', 'nodes', 'pruned', 'nodes_saved', 'heuristic_cutoffs', 'win_rate', 'iterations', 'elapsed_ms', 'destinations', 'attractions', 'cuisines', 'hotels']
  const metrics = keys.filter((key) => data[key] !== undefined).map((key) => [key, data[key]])
  if (metrics.length === 0) return null
  return (
    <section className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
      {metrics.slice(0, 4).map(([key, value]) => (
        <div key={key} className={`rounded-md ${script.tint} p-3`}>
          <p className={`text-xs font-black uppercase ${script.text}`}>{formatLabel(key)}</p>
          <p className="mt-1 text-xl font-black">{formatValue(value)}</p>
        </div>
      ))}
    </section>
  )
}

function BoardPanel({ data, script }) {
  if (!Array.isArray(data.board)) return null
  return (
    <section className="grid gap-4 rounded-md border border-slate-200 bg-slate-50 p-4 md:grid-cols-[180px_1fr]">
      <div className="grid aspect-square grid-cols-3 gap-2">
        {data.board.map((cell, index) => (
          <div
            key={index}
            className={`flex items-center justify-center rounded-md text-2xl font-black ${
              data.best_move === index ? `${script.color} text-white` : 'bg-white text-slate-900'
            }`}
          >
            {cell === 1 ? 'X' : cell === -1 ? 'O' : index}
          </div>
        ))}
      </div>
      <div>
        <p className="text-sm font-black">Board view</p>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          The highlighted square is the move selected by the current algorithm.
        </p>
        {data.algorithm ? <p className="mt-3 text-sm font-black text-slate-800">{data.algorithm}</p> : null}
        {data.complexity ? <p className="mt-2 font-mono text-xs text-slate-500">{data.complexity}</p> : null}
      </div>
    </section>
  )
}

function SearchBars({ data, script }) {
  const rows = [
    ['Nodes', data.nodes],
    ['Pruned', data.pruned],
    ['Saved', data.nodes_saved],
    ['Cutoffs', data.heuristic_cutoffs],
  ].filter(([, value]) => typeof value === 'number')
  if (rows.length === 0) return null
  const max = Math.max(...rows.map(([, value]) => value), 1)
  return (
    <section className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-black">Search effort</p>
      <div className="mt-3 grid gap-2">
        {rows.map(([label, value]) => (
          <div key={label} className="grid grid-cols-[78px_1fr_72px] items-center gap-3 text-sm">
            <span className="font-bold text-slate-600">{label}</span>
            <div className="h-2 overflow-hidden rounded-full bg-slate-100">
              <div className={`h-full rounded-full ${script.color}`} style={{ width: `${Math.max(6, (value / max) * 100)}%` }} />
            </div>
            <span className="text-right font-black">{formatValue(value)}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function MctsPanel({ stats, script }) {
  if (!Array.isArray(stats) || stats.length === 0) return null
  const maxVisits = Math.max(...stats.map((item) => item.visits), 1)
  return (
    <section className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-black">MCTS move confidence</p>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        {stats.slice(0, 6).map((item) => (
          <div key={item.move} className="rounded-md bg-slate-50 p-3">
            <div className="flex items-center justify-between text-sm">
              <span className="font-black">Move {item.move}</span>
              <span className="font-mono text-xs text-slate-500">{item.visits}</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-white">
              <div className={`h-full rounded-full ${script.color}`} style={{ width: `${(item.visits / maxVisits) * 100}%` }} />
            </div>
            <p className="mt-2 text-xs font-bold text-slate-600">Win rate {formatValue(item.win_rate)}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function RankedScores({ scores, weights }) {
  if (!Array.isArray(scores) || scores.length === 0) return null
  return (
    <section className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-black">Best destination matches</p>
      <div className="mt-3 space-y-2">
        {scores.map((item) => (
          <div key={item.destination} className="grid grid-cols-[116px_1fr_48px] items-center gap-3 text-sm">
            <span className="font-black text-slate-700">{item.destination}</span>
            <div className="h-2 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full bg-[#24B47E]" style={{ width: `${Math.min(100, item.score)}%` }} />
            </div>
            <span className="text-right font-black">{item.score}</span>
          </div>
        ))}
      </div>
      {weights ? <p className="mt-3 text-xs font-bold text-slate-500">Scored on budget, activity, diet, season, and region.</p> : null}
    </section>
  )
}

function ToolCatalog({ tools }) {
  if (!Array.isArray(tools) || tools.length === 0) return null
  return (
    <section className="grid gap-2 md:grid-cols-2">
      {tools.slice(0, 6).map((group) => (
        <article key={group.category || group.tool} className="rounded-md border border-slate-200 bg-white p-3">
          <p className="text-sm font-black">{group.category || group.tool}</p>
          <p className="mt-1 text-sm leading-6 text-slate-600">{group.use_case || group.role}</p>
          {Array.isArray(group.tools) ? <p className="mt-2 text-xs font-bold text-slate-500">{group.tools.join(' / ')}</p> : null}
        </article>
      ))}
    </section>
  )
}

function KnowledgeGraph({ edges }) {
  if (!Array.isArray(edges) || edges.length === 0) return null
  const nodes = Array.from(new Set(edges.flatMap((edge) => [edge.source, edge.target]))).slice(0, 24)
  const positions = radialPositions(nodes, 360, 220, 138, 76)
  return (
    <section className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-black">Knowledge graph snapshot</p>
      <svg viewBox="0 0 360 220" className="mt-3 h-[240px] w-full rounded-md bg-[#E9FFF5]">
        {edges.map((edge) => positions[edge.source] && positions[edge.target] ? (
          <line key={`${edge.source}-${edge.target}-${edge.relation}`} x1={positions[edge.source].x} y1={positions[edge.source].y} x2={positions[edge.target].x} y2={positions[edge.target].y} stroke="#94a3b8" strokeWidth="1" />
        ) : null)}
        {nodes.map((node) => (
          <g key={node}>
            <circle cx={positions[node].x} cy={positions[node].y} r={isDestinationName(node) ? 13 : 8} fill={isDestinationName(node) ? '#24B47E' : '#111827'} />
            <text x={positions[node].x} y={positions[node].y + 22} textAnchor="middle" className="fill-slate-700 text-[8px] font-bold">{shortLabel(node)}</text>
          </g>
        ))}
      </svg>
    </section>
  )
}

function BayesianGraph({ nodes, edges, labels }) {
  if (!Array.isArray(nodes) || !Array.isArray(edges)) return null
  const positions = {
    A: { x: 50, y: 55 }, S: { x: 50, y: 150 }, T: { x: 150, y: 55 }, L: { x: 150, y: 125 },
    B: { x: 150, y: 195 }, E: { x: 245, y: 90 }, X: { x: 325, y: 55 }, D: { x: 325, y: 160 },
  }
  return (
    <section className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-black">Bayesian network</p>
      <svg viewBox="0 0 380 225" className="mt-3 h-[245px] w-full rounded-md bg-[#FFF0F1]">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" fill="#64748b" />
          </marker>
        </defs>
        {edges.map(([source, target]) => (
          <line key={`${source}-${target}`} x1={positions[source].x} y1={positions[source].y} x2={positions[target].x} y2={positions[target].y} stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrow)" />
        ))}
        {nodes.map((node) => (
          <g key={node}>
            <circle cx={positions[node].x} cy={positions[node].y} r="18" fill="#FF5A5F" />
            <text x={positions[node].x} y={positions[node].y + 5} textAnchor="middle" className="fill-white text-sm font-black">{node}</text>
            <text x={positions[node].x} y={positions[node].y + 33} textAnchor="middle" className="fill-slate-700 text-[9px] font-bold">{labels?.[node] ?? node}</text>
          </g>
        ))}
      </svg>
    </section>
  )
}

function ProbabilityGrid({ values }) {
  if (!values || typeof values !== 'object') return null
  return (
    <section className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
      {Object.entries(values).map(([key, value]) => (
        <div key={key} className="rounded-md bg-[#FFF0F1] p-3">
          <p className="text-xs font-black text-[#B83236]">P({key}=yes)</p>
          <p className="mt-1 text-xl font-black">{formatValue(value)}</p>
        </div>
      ))}
    </section>
  )
}

function ComparisonTable({ rows }) {
  if (!Array.isArray(rows) || rows.length === 0) return null
  const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
  return (
    <section className="overflow-hidden rounded-md border border-slate-200 bg-white">
      <div className="border-b border-slate-200 p-3 text-sm font-black">Comparison</div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-left text-sm">
          <thead className="bg-slate-50 text-xs uppercase text-slate-500">
            <tr>{keys.map((key) => <th key={key} className="px-3 py-2">{formatLabel(key)}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index} className="border-t border-slate-100">
                {keys.map((key) => <td key={key} className="px-3 py-2 font-semibold text-slate-700">{formatReadable(row[key])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function TourPlan({ data }) {
  if (!data.destination || data.total_estimated_cost === undefined) return null
  return (
    <section className="rounded-md bg-[#E9FFF5] p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-black uppercase text-[#147A56]">Recommended plan</p>
          <h4 className="mt-1 text-2xl font-black">{data.destination}</h4>
          <p className="mt-1 text-sm text-slate-700">{data.hotel}</p>
        </div>
        <div className="text-left sm:text-right">
          <p className="text-xs font-black uppercase text-[#147A56]">Estimated total</p>
          <p className="text-2xl font-black">${formatValue(data.total_estimated_cost)}</p>
        </div>
      </div>
    </section>
  )
}

function StringList({ title, items }) {
  if (!Array.isArray(items) || items.length === 0 || items.some((item) => typeof item !== 'string')) return null
  return (
    <section className="rounded-md border border-slate-200 bg-white p-3">
      <p className="text-sm font-black">{title}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {items.map((item) => <span key={item} className="rounded-md bg-slate-100 px-3 py-2 text-sm font-bold text-slate-700">{item}</span>)}
      </div>
    </section>
  )
}

function radialPositions(nodes, width, height, radiusX, radiusY) {
  const centerX = width / 2
  const centerY = height / 2
  return Object.fromEntries(nodes.map((node, index) => {
    const angle = (index / nodes.length) * Math.PI * 2
    return [node, { x: centerX + Math.cos(angle) * radiusX, y: centerY + Math.sin(angle) * radiusY }]
  }))
}

function isDestinationName(name) {
  return ['Paris', 'Tokyo', 'Rajasthan', 'Santorini', 'Kyoto', 'Bali', 'New York', 'Cape Town', 'Machu Picchu', 'Dubai', 'Barcelona', 'Kerala'].includes(name)
}

function shortLabel(value) {
  return String(value).length > 14 ? `${String(value).slice(0, 13)}.` : String(value)
}

function formatLabel(key) {
  return String(key).replaceAll('_', ' ')
}

function formatValue(value) {
  if (value === null || value === undefined) return 'n/a'
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 4 })
  return String(value)
}

function formatReadable(value) {
  if (Array.isArray(value)) return value.map(formatReadable).join(', ')
  if (value && typeof value === 'object') return Object.entries(value).map(([key, val]) => `${key}=${formatReadable(val)}`).join(', ')
  return formatValue(value)
}

export default App2
