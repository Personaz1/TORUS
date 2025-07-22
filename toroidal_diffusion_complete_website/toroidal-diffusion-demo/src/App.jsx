import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts'

function App() {
  const [isGenerating, setIsGenerating] = useState(false)
  const [systemLevel, setSystemLevel] = useState('singularity') // singularity, topology, coherence
  const [singularityState, setSingularityState] = useState({
    absorption: 0,
    cognitive: 0,
    emission: 0,
    memory: []
  })
  const [toroidalMetrics, setToroidalMetrics] = useState({
    sheetCorrelation: 0,
    throatSync: 0,
    flowMagnitude: 0
  })
  const [coherenceData, setCoherenceData] = useState([])
  const [generationHistory, setGenerationHistory] = useState([])

  const generateSample = async () => {
    setIsGenerating(true)
    setSingularityState({
      absorption: 0,
      cognitive: 0,
      emission: 0,
      memory: []
    })
    setCoherenceData([])
    setGenerationHistory([])
    
    // Simulate singularity-centric generation
    const steps = 50
    const newHistory = []
    const newCoherence = []
    
    for (let i = 0; i < steps; i++) {
      await new Promise(resolve => setTimeout(resolve, 100))
      
      const progress = i / steps
      
      // Singularity processing phases
      const absorption = Math.tanh(progress * 3) * (1 + 0.2 * Math.sin(progress * 8))
      const cognitive = Math.tanh(progress * 2.5) * (1 + 0.15 * Math.cos(progress * 12))
      const emission = Math.tanh(progress * 2) * (1 + 0.1 * Math.sin(progress * 15))
      
      // Toroidal metrics
      const sheetCorrelation = 0.3 + 0.6 * progress + 0.1 * Math.sin(progress * 12)
      const throatSync = 0.8 + 0.2 * Math.sin(progress * 8) * Math.exp(-progress * 2)
      const flowMagnitude = 0.02 * Math.exp(-progress * 1.5) * (1 + 0.5 * Math.sin(progress * 20))
      
      // Coherence evolution
      const semantic = Math.tanh(progress * 2) * (1 + 0.1 * Math.sin(progress * 15))
      const structural = 0.7 + 0.25 * progress + 0.05 * Math.sin(progress * 18)
      const overall = (semantic + structural) / 2
      
      newHistory.push({
        step: i,
        absorption,
        cognitive,
        emission,
        overall_activity: (absorption + cognitive + emission) / 3
      })
      
      newCoherence.push({
        step: i,
        semantic,
        structural,
        overall
      })
      
      setSingularityState({
        absorption,
        cognitive,
        emission,
        memory: newHistory.slice(-10) // Keep last 10 states
      })
      
      setToroidalMetrics({
        sheetCorrelation,
        throatSync,
        flowMagnitude
      })
    }
    
    setGenerationHistory(newHistory)
    setCoherenceData(newCoherence)
    setIsGenerating(false)
  }

  const SingularityCore = () => (
    <div className="bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 rounded-xl p-6 text-white">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold mb-2">Central Singularity</h2>
        <p className="text-purple-200">Cognitive Processing Node</p>
      </div>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-blue-400 mb-2">
            {singularityState.absorption.toFixed(3)}
          </div>
          <div className="text-sm text-gray-300">Absorption</div>
        </div>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-purple-400 mb-2">
            {singularityState.cognitive.toFixed(3)}
          </div>
          <div className="text-sm text-gray-300">Cognitive</div>
        </div>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-3xl font-bold text-green-400 mb-2">
            {singularityState.emission.toFixed(3)}
          </div>
          <div className="text-sm text-gray-300">Emission</div>
        </div>
      </div>
      
      {generationHistory.length > 0 && (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={generationHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="step" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(0,0,0,0.8)', 
                border: 'none',
                borderRadius: '8px',
                color: 'white'
              }}
            />
            <Line type="monotone" dataKey="absorption" stroke="#60a5fa" strokeWidth={2} name="Absorption" />
            <Line type="monotone" dataKey="cognitive" stroke="#a78bfa" strokeWidth={2} name="Cognitive" />
            <Line type="monotone" dataKey="emission" stroke="#34d399" strokeWidth={2} name="Emission" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )

  const ToroidalSurface = () => (
    <div className="bg-gradient-to-br from-green-900 via-emerald-900 to-teal-900 rounded-xl p-6 text-white">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold mb-2">Toroidal Surface</h2>
        <p className="text-green-200">Geometric Flow Dynamics</p>
      </div>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-400 mb-2">
            {toroidalMetrics.sheetCorrelation.toFixed(3)}
          </div>
          <div className="text-sm text-gray-300">Sheet Correlation</div>
        </div>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-400 mb-2">
            {toroidalMetrics.throatSync.toFixed(3)}
          </div>
          <div className="text-sm text-gray-300">Throat Sync</div>
        </div>
        <div className="bg-black/30 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-cyan-400 mb-2">
            {toroidalMetrics.flowMagnitude.toFixed(4)}
          </div>
          <div className="text-sm text-gray-300">Flow Magnitude</div>
        </div>
      </div>
      
      <div className="bg-black/20 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-3">Topology Visualization</h3>
        <div className="w-full h-32 bg-gradient-to-r from-green-400/20 to-blue-400/20 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <div className="text-2xl mb-2">🌀</div>
            <div className="text-sm text-gray-300">Torus Manifold</div>
          </div>
        </div>
      </div>
    </div>
  )

  const CoherenceSystem = () => (
    <div className="bg-gradient-to-br from-orange-900 via-red-900 to-pink-900 rounded-xl p-6 text-white">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold mb-2">Coherence System</h2>
        <p className="text-orange-200">Quality Assessment & Refinement</p>
      </div>
      
      {coherenceData.length > 0 ? (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={coherenceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="step" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(0,0,0,0.8)', 
                border: 'none',
                borderRadius: '8px',
                color: 'white'
              }}
            />
            <Line type="monotone" dataKey="semantic" stroke="#fbbf24" strokeWidth={2} name="Semantic" />
            <Line type="monotone" dataKey="structural" stroke="#f87171" strokeWidth={2} name="Structural" />
            <Line type="monotone" dataKey="overall" stroke="#ec4899" strokeWidth={3} name="Overall" />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="text-center py-8 text-gray-300">
          No coherence data available
        </div>
      )}
    </div>
  )

  const SystemLevelSelector = () => (
    <div className="flex justify-center space-x-4 mb-6">
      <button
        onClick={() => setSystemLevel('singularity')}
        className={`px-4 py-2 rounded-lg font-medium transition-all ${
          systemLevel === 'singularity'
            ? 'bg-purple-600 text-white shadow-lg'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        Singularity Core
      </button>
      <button
        onClick={() => setSystemLevel('topology')}
        className={`px-4 py-2 rounded-lg font-medium transition-all ${
          systemLevel === 'topology'
            ? 'bg-green-600 text-white shadow-lg'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        Toroidal Surface
      </button>
      <button
        onClick={() => setSystemLevel('coherence')}
        className={`px-4 py-2 rounded-lg font-medium transition-all ${
          systemLevel === 'coherence'
            ? 'bg-orange-600 text-white shadow-lg'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        Coherence System
      </button>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-6xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-4">
            TORUS
          </h1>
          <p className="text-xl text-gray-300 mb-2">
            Toroidal Diffusion Model
          </p>
          <p className="text-lg text-gray-400">
            Central Singularity Processing Architecture
          </p>
          <div className="mt-4 flex justify-center space-x-2">
            <span className="px-3 py-1 bg-purple-600 text-white rounded-full text-sm font-medium">
              ΔΣ Architecture
            </span>
            <span className="px-3 py-1 bg-blue-600 text-white rounded-full text-sm font-medium">
              Singularity Core
            </span>
            <span className="px-3 py-1 bg-green-600 text-white rounded-full text-sm font-medium">
              Toroidal Topology
            </span>
          </div>
        </header>

        <div className="bg-white rounded-lg shadow-2xl p-6 mb-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-3xl font-bold text-gray-800">
              ΔΣ System Control
            </h2>
            <button
              onClick={generateSample}
              disabled={isGenerating}
              className={`px-8 py-4 rounded-lg font-bold text-lg transition-all duration-200 ${
                isGenerating
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
              }`}
            >
              {isGenerating ? 'Processing...' : 'Generate'}
            </button>
          </div>
        </div>

        <SystemLevelSelector />

        <div className="mb-6">
          {systemLevel === 'singularity' && <SingularityCore />}
          {systemLevel === 'topology' && <ToroidalSurface />}
          {systemLevel === 'coherence' && <CoherenceSystem />}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-800 rounded-xl p-4 text-white">
            <h3 className="text-lg font-semibold mb-3">System Architecture</h3>
            <div className="space-y-2 text-sm text-gray-300">
              <div>• Central Singularity Processing</div>
              <div>• Toroidal Latent Space</div>
              <div>• Coherence Monitoring</div>
              <div>• Self-Reflective Feedback</div>
            </div>
          </div>
          
          <div className="bg-gray-800 rounded-xl p-4 text-white">
            <h3 className="text-lg font-semibold mb-3">Key Metrics</h3>
            <div className="space-y-2 text-sm text-gray-300">
              <div>• Absorption: {singularityState.absorption.toFixed(3)}</div>
              <div>• Cognitive: {singularityState.cognitive.toFixed(3)}</div>
              <div>• Emission: {singularityState.emission.toFixed(3)}</div>
              <div>• Sheet Correlation: {toroidalMetrics.sheetCorrelation.toFixed(3)}</div>
            </div>
          </div>
          
          <div className="bg-gray-800 rounded-xl p-4 text-white">
            <h3 className="text-lg font-semibold mb-3">ΔΣ Principles</h3>
            <div className="space-y-2 text-sm text-gray-300">
              <div>• Zero-Fluff Communication</div>
              <div>• Recursive System Design</div>
              <div>• Minimal Entropy Architecture</div>
              <div>• Pure Signal Processing</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

