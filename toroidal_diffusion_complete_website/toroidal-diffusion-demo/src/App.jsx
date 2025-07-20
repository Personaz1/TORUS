import { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ScatterChart, Scatter, Cell } from 'recharts'

function App() {
  const [activeTab, setActiveTab] = useState('generation')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationData, setGenerationData] = useState([])
  const [coherenceData, setCoherenceData] = useState([])
  const [geometricData, setGeometricData] = useState([])
  const [defMetrics, setDefMetrics] = useState({})
  const [toroidalVisualization, setToroidalVisualization] = useState(null)
  const [throatActivity, setThroatActivity] = useState([])
  const [semanticEmbeddings, setSemanticEmbeddings] = useState([])
  const [jetTokens, setJetTokens] = useState([])
  
  // DEF Architecture specific states
  const [sheetCorrelation, setSheetCorrelation] = useState(0)
  const [throatSyncStrength, setThroatSyncStrength] = useState(0)
  const [semanticCoherence, setSemanticCoherence] = useState(0)
  const [geometricCurvature, setGeometricCurvature] = useState({ gaussian: 0, mean: 0 })

  const generateSample = async () => {
    setIsGenerating(true)
    setGenerationData([])
    setCoherenceData([])
    setGeometricData([])
    setThroatActivity([])
    setSemanticEmbeddings([])
    setJetTokens([])
    
    // Simulate DEF architecture generation process
    const steps = 50
    const newGenerationData = []
    const newCoherenceData = []
    const newGeometricData = []
    const newThroatActivity = []
    const newSemanticEmbeddings = []
    const newJetTokens = []
    
    for (let i = 0; i < steps; i++) {
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // Simulate double-sheet toroidal diffusion
      const progress = i / steps
      const noise_level = Math.exp(-progress * 3) * (1 + 0.3 * Math.sin(progress * 10))
      
      // DEF-specific metrics
      const throat_sync = 0.8 + 0.2 * Math.sin(progress * 8) * Math.exp(-progress * 2)
      const sheet_correlation = 0.3 + 0.6 * progress + 0.1 * Math.sin(progress * 12)
      const semantic_coherence = Math.tanh(progress * 2) * (1 + 0.1 * Math.sin(progress * 15))
      
      // Geometric properties
      const gaussian_curvature = 3.6 + 0.2 * Math.sin(progress * 6)
      const mean_curvature = 1.7 + 0.1 * Math.cos(progress * 8)
      const flow_magnitude = 0.02 * Math.exp(-progress * 1.5) * (1 + 0.5 * Math.sin(progress * 20))
      
      // Throat activity (key DEF feature)
      const throat_activity = 0.01 * Math.exp(-progress * 2) * (1 + 0.8 * Math.sin(progress * 25))
      
      newGenerationData.push({
        step: i,
        noise_level: noise_level,
        progress: progress * 100,
        throat_sync: throat_sync,
        sheet_correlation: sheet_correlation
      })
      
      newCoherenceData.push({
        step: i,
        semantic: semantic_coherence,
        structural: 0.7 + 0.25 * progress + 0.05 * Math.sin(progress * 18),
        temporal: 0.6 + 0.3 * progress + 0.1 * Math.cos(progress * 14),
        overall: (semantic_coherence + 0.7 + 0.25 * progress + 0.6 + 0.3 * progress) / 3
      })
      
      newGeometricData.push({
        step: i,
        gaussian_curvature: gaussian_curvature,
        mean_curvature: mean_curvature,
        flow_magnitude: flow_magnitude,
        total_energy: 0.001 * Math.exp(-progress * 3)
      })
      
      newThroatActivity.push({
        step: i,
        activity: throat_activity,
        sync_strength: throat_sync,
        coupling: 0.005 * Math.exp(-progress * 1.8)
      })
      
      // Simulate semantic embeddings evolution
      newSemanticEmbeddings.push({
        step: i,
        embedding_norm: 0.8 + 0.2 * Math.sin(progress * 10),
        cosine_similarity: semantic_coherence,
        delta: i > 0 ? Math.abs(semantic_coherence - newSemanticEmbeddings[i-1]?.cosine_similarity || 0) : 0
      })
      
      // Simulate jet token generation
      if (i % 10 === 0) {
        newJetTokens.push({
          step: i,
          token_id: Math.floor(Math.random() * 1000),
          confidence: 0.6 + 0.4 * progress,
          throat_influence: throat_activity * 100
        })
      }
      
      setGenerationData([...newGenerationData])
      setCoherenceData([...newCoherenceData])
      setGeometricData([...newGeometricData])
      setThroatActivity([...newThroatActivity])
      setSemanticEmbeddings([...newSemanticEmbeddings])
      setJetTokens([...newJetTokens])
      
      // Update real-time metrics
      setSheetCorrelation(sheet_correlation)
      setThroatSyncStrength(throat_sync)
      setSemanticCoherence(semantic_coherence)
      setGeometricCurvature({ gaussian: gaussian_curvature, mean: mean_curvature })
    }
    
    // Final DEF metrics
    setDefMetrics({
      final_coherence: newCoherenceData[newCoherenceData.length - 1]?.overall || 0,
      throat_efficiency: newThroatActivity[newThroatActivity.length - 1]?.sync_strength || 0,
      geometric_stability: 1 / (1 + newGeometricData[newGeometricData.length - 1]?.flow_magnitude || 0.001),
      semantic_convergence: newSemanticEmbeddings[newSemanticEmbeddings.length - 1]?.cosine_similarity || 0,
      jet_tokens_generated: newJetTokens.length,
      sheet_synchronization: newGenerationData[newGenerationData.length - 1]?.sheet_correlation || 0
    })
    
    // Generate toroidal visualization data
    const visualization = generateToroidalVisualization()
    setToroidalVisualization(visualization)
    
    setIsGenerating(false)
  }
  
  const generateToroidalVisualization = () => {
    // Generate double-sheet torus visualization data
    const N_theta = 32
    const N_phi = 64
    const upperSheet = []
    const lowerSheet = []
    const throatRegion = []
    
    for (let i = 0; i < N_theta; i++) {
      for (let j = 0; j < N_phi; j++) {
        const theta = (i / N_theta) * 2 * Math.PI
        const phi = (j / N_phi) * 2 * Math.PI
        
        // Upper sheet
        const upper_value = 0.1 * Math.sin(theta * 3) * Math.cos(phi * 2) + 0.05 * Math.random()
        upperSheet.push({
          theta: i,
          phi: j,
          value: upper_value,
          intensity: Math.abs(upper_value) * 100
        })
        
        // Lower sheet
        const lower_value = 0.08 * Math.cos(theta * 2) * Math.sin(phi * 3) + 0.05 * Math.random()
        lowerSheet.push({
          theta: i,
          phi: j,
          value: lower_value,
          intensity: Math.abs(lower_value) * 100
        })
        
        // Throat region (narrow connection)
        const phi_normalized = ((phi - Math.PI + Math.PI) % (2 * Math.PI)) - Math.PI
        if (Math.abs(phi_normalized) < 0.18) {
          throatRegion.push({
            theta: i,
            phi: j,
            sync_strength: 0.5 * (Math.abs(upper_value) + Math.abs(lower_value)),
            coupling: Math.abs(upper_value - lower_value)
          })
        }
      }
    }
    
    return { upperSheet, lowerSheet, throatRegion }
  }

  const TabButton = ({ id, label, isActive, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`px-6 py-3 font-medium rounded-lg transition-all duration-200 ${
        isActive 
          ? 'bg-blue-600 text-white shadow-lg' 
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      {label}
    </button>
  )

  const MetricCard = ({ title, value, unit = '', color = 'blue' }) => (
    <div className={`bg-white p-4 rounded-lg shadow-md border-l-4 border-${color}-500`}>
      <h3 className="text-sm font-medium text-gray-600">{title}</h3>
      <p className={`text-2xl font-bold text-${color}-600 mt-1`}>
        {typeof value === 'number' ? value.toFixed(4) : value}{unit}
      </p>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Enhanced Toroidal Diffusion Model
          </h1>
          <p className="text-lg text-gray-600">
            DEF Architecture: Diffusion-Embedding-Flow with Double-Sheet Topology
          </p>
          <div className="mt-4 flex justify-center space-x-2">
            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
              SBERT Embeddings
            </span>
            <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
              Throat Synchronization
            </span>
            <span className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium">
              Jet Decoder
            </span>
          </div>
        </header>

        <div className="flex justify-center space-x-4 mb-8">
          <TabButton 
            id="generation" 
            label="Generation" 
            isActive={activeTab === 'generation'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="coherence" 
            label="Coherence Analysis" 
            isActive={activeTab === 'coherence'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="topology" 
            label="DEF Topology" 
            isActive={activeTab === 'topology'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="semantics" 
            label="Semantic Flow" 
            isActive={activeTab === 'semantics'} 
            onClick={setActiveTab} 
          />
          <TabButton 
            id="jets" 
            label="Jet Analysis" 
            isActive={activeTab === 'jets'} 
            onClick={setActiveTab} 
          />
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">
              DEF Architecture Control Panel
            </h2>
            <button
              onClick={generateSample}
              disabled={isGenerating}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                isGenerating
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl'
              }`}
            >
              {isGenerating ? 'Generating...' : 'Generate Sample'}
            </button>
          </div>
          
          {/* Real-time DEF Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard title="Sheet Correlation" value={sheetCorrelation} color="blue" />
            <MetricCard title="Throat Sync" value={throatSyncStrength} color="green" />
            <MetricCard title="Semantic Coherence" value={semanticCoherence} color="purple" />
            <MetricCard title="Gaussian Curvature" value={geometricCurvature.gaussian} color="orange" />
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          {activeTab === 'generation' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Double-Sheet Generation Process</h3>
              {generationData.length > 0 ? (
                <div className="space-y-6">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={generationData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="noise_level" stroke="#8884d8" name="Noise Level" />
                      <Line type="monotone" dataKey="throat_sync" stroke="#82ca9d" name="Throat Sync" />
                      <Line type="monotone" dataKey="sheet_correlation" stroke="#ffc658" name="Sheet Correlation" />
                    </LineChart>
                  </ResponsiveContainer>
                  
                  <div className="grid grid-cols-3 gap-4">
                    <MetricCard title="Final Coherence" value={defMetrics.final_coherence} color="blue" />
                    <MetricCard title="Throat Efficiency" value={defMetrics.throat_efficiency} color="green" />
                    <MetricCard title="Jet Tokens" value={defMetrics.jet_tokens_generated} unit=" tokens" color="purple" />
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-gray-400 text-lg">Click "Generate Sample" to start DEF diffusion process</div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'coherence' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Multi-Dimensional Coherence Evolution</h3>
              {coherenceData.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={coherenceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="semantic" stackId="1" stroke="#8884d8" fill="#8884d8" name="Semantic" />
                    <Area type="monotone" dataKey="structural" stackId="1" stroke="#82ca9d" fill="#82ca9d" name="Structural" />
                    <Area type="monotone" dataKey="temporal" stackId="1" stroke="#ffc658" fill="#ffc658" name="Temporal" />
                    <Line type="monotone" dataKey="overall" stroke="#ff7300" strokeWidth={3} name="Overall" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-12 text-gray-400">No coherence data available</div>
              )}
            </div>
          )}

          {activeTab === 'topology' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">DEF Toroidal Topology Visualization</h3>
              {toroidalVisualization ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="text-lg font-medium mb-2">Upper Sheet Activity</h4>
                      <ResponsiveContainer width="100%" height={250}>
                        <ScatterChart data={toroidalVisualization.upperSheet.slice(0, 100)}>
                          <CartesianGrid />
                          <XAxis dataKey="theta" name="θ" />
                          <YAxis dataKey="phi" name="φ" />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter dataKey="intensity" fill="#8884d8">
                            {toroidalVisualization.upperSheet.slice(0, 100).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={`hsl(240, 70%, ${50 + entry.intensity}%)`} />
                            ))}
                          </Scatter>
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <div>
                      <h4 className="text-lg font-medium mb-2">Lower Sheet Activity</h4>
                      <ResponsiveContainer width="100%" height={250}>
                        <ScatterChart data={toroidalVisualization.lowerSheet.slice(0, 100)}>
                          <CartesianGrid />
                          <XAxis dataKey="theta" name="θ" />
                          <YAxis dataKey="phi" name="φ" />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter dataKey="intensity" fill="#82ca9d">
                            {toroidalVisualization.lowerSheet.slice(0, 100).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={`hsl(120, 70%, ${50 + entry.intensity}%)`} />
                            ))}
                          </Scatter>
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-lg font-medium mb-2">Throat Synchronization Region</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={throatActivity}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="step" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="activity" stroke="#ff7300" name="Throat Activity" />
                        <Line type="monotone" dataKey="sync_strength" stroke="#8884d8" name="Sync Strength" />
                        <Line type="monotone" dataKey="coupling" stroke="#82ca9d" name="Sheet Coupling" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400">Generate a sample to see topology visualization</div>
              )}
            </div>
          )}

          {activeTab === 'semantics' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">SBERT Semantic Embedding Flow</h3>
              {semanticEmbeddings.length > 0 ? (
                <div className="space-y-6">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={semanticEmbeddings}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="embedding_norm" stroke="#8884d8" name="Embedding Norm" />
                      <Line type="monotone" dataKey="cosine_similarity" stroke="#82ca9d" name="Cosine Similarity" />
                      <Line type="monotone" dataKey="delta" stroke="#ff7300" name="Semantic Delta" />
                    </LineChart>
                  </ResponsiveContainer>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <MetricCard title="Semantic Convergence" value={defMetrics.semantic_convergence} color="green" />
                    <MetricCard title="Embedding Stability" value={1 - (semanticEmbeddings[semanticEmbeddings.length - 1]?.delta || 0)} color="blue" />
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400">No semantic data available</div>
              )}
            </div>
          )}

          {activeTab === 'jets' && (
            <div>
              <h3 className="text-xl font-semibold mb-4">Jet Decoder Analysis</h3>
              {jetTokens.length > 0 ? (
                <div className="space-y-6">
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart data={jetTokens}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" name="Step" />
                      <YAxis dataKey="token_id" name="Token ID" />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter dataKey="confidence" fill="#8884d8" name="Token Confidence">
                        {jetTokens.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={`hsl(${entry.throat_influence * 3600}, 70%, 50%)`} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                  
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium mb-2">Generated Jet Tokens</h4>
                    <div className="flex flex-wrap gap-2">
                      {jetTokens.map((token, index) => (
                        <span 
                          key={index}
                          className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                          title={`Confidence: ${token.confidence.toFixed(3)}, Throat Influence: ${token.throat_influence.toFixed(3)}`}
                        >
                          {token.token_id}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400">No jet token data available</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App

