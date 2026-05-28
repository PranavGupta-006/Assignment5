import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { spawn } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')

const scriptMap = {
  q1: {
    file: 'q1_game_search.py',
    title: 'Q1 Game Search',
  },
  q2: {
    file: 'q2_travel_planner.py',
    title: 'Q2 Travel Planner',
  },
  q3: {
    file: 'q3_bayesian_networks.py',
    title: 'Q3 Bayesian Networks',
  },
  q4: {
    file: 'q4_knowledge_base.py',
    title: 'Q4 Knowledge Base',
  },
}

function scriptApiPlugin() {
  return {
    name: 'assignment-script-api',
    configureServer(server) {
      server.middlewares.use('/api/scripts', (_req, res) => {
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(Object.entries(scriptMap).map(([id, script]) => ({
          id,
          title: script.title,
          file: script.file,
        }))))
      })

      server.middlewares.use('/api/run', (req, res) => {
        const id = req.url?.split('?')[0]?.replace(/^\/+/, '')
        const script = scriptMap[id]

        if (!script) {
          res.statusCode = 404
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ error: 'Unknown script id' }))
          return
        }

        res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8')
        res.setHeader('Cache-Control', 'no-cache')
        res.setHeader('Connection', 'keep-alive')

        const child = spawn('python3', [resolve(rootDir, script.file), '--steps'], {
          cwd: rootDir,
          stdio: ['ignore', 'pipe', 'pipe'],
        })

        req.on('close', () => {
          if (!child.killed) child.kill()
        })

        child.stdout.on('data', (chunk) => {
          res.write(chunk)
        })

        child.stderr.on('data', (chunk) => {
          res.write(JSON.stringify({
            title: 'Script stderr',
            status: 'error',
            detail: chunk.toString(),
            data: {},
          }) + '\n')
        })

        child.on('close', (code) => {
          if (code !== 0) {
            res.write(JSON.stringify({
              title: `${script.title} exited with code ${code}`,
              status: 'error',
              detail: 'The Python process ended before completing successfully.',
              data: { code },
            }) + '\n')
          }
          res.end()
        })
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), scriptApiPlugin()],
})
