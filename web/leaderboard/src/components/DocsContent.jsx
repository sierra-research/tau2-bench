import { useState, useEffect } from 'react'
import './DocsContent.css'

const DocsContent = ({ domain }) => {
  const [policyContent, setPolicyContent] = useState('')
  const [toolsData, setToolsData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Domain documentation content
  const domainDocs = {
    airline: {
      title: "Airline Agent Policy",
      description: "Documentation for the airline customer service domain",
      policyPath: "task-data/domains/airline/policy.md"
    },
    retail: {
      title: "Retail Agent Policy", 
      description: "Documentation for the retail customer service domain",
      policyPath: "task-data/domains/retail/policy.md"
    },
    telecom: {
      title: "Telecom Agent Policy",
      description: "Documentation for the telecom customer service domain", 
      policyPath: "task-data/domains/telecom/main_policy.md"
    },
    mock: {
      title: "Mock Domain Policy",
      description: "Documentation for the mock testing domain",
      policyPath: "task-data/domains/mock/policy.md"
    }
  }

  useEffect(() => {
    const loadContent = async () => {
      setLoading(true)
      setError(null)
      
      try {
        // Load policy content
        const policyResponse = await fetch(`${import.meta.env.BASE_URL}${domainDocs[domain].policyPath}`)
        if (!policyResponse.ok) {
          throw new Error(`Failed to load policy: ${policyResponse.statusText}`)
        }
        const policyContent = await policyResponse.text()
        setPolicyContent(policyContent)

        // Load tools data
        const toolsResponse = await fetch(`${import.meta.env.BASE_URL}task-data/tools-data.json`)
        if (!toolsResponse.ok) {
          throw new Error(`Failed to load tools data: ${toolsResponse.statusText}`)
        }
        const toolsData = await toolsResponse.json()
        setToolsData(toolsData)
      } catch (err) {
        setError(err.message)
        console.error('Error loading content:', err)
      } finally {
        setLoading(false)
      }
    }

    loadContent()
  }, [domain])

  // Convert markdown to HTML (basic implementation)
  const renderMarkdown = (markdown) => {
    // Split into lines to process properly
    const lines = markdown.split('\n')
    let html = ''
    let inList = false
    
    for (let i = 0; i < lines.length; i++) {
      let line = lines[i]
      
             // Handle headers
       if (line.startsWith('### ')) {
         if (inList) { html += `</${inList}>`; inList = false; }
         html += `<h3>${line.substring(4)}</h3>`
       } else if (line.startsWith('## ')) {
         if (inList) { html += `</${inList}>`; inList = false; }
         html += `<h2>${line.substring(3)}</h2>`
       } else if (line.startsWith('# ')) {
         if (inList) { html += `</${inList}>`; inList = false; }
         html += `<h1>${line.substring(2)}</h1>`
       }
      // Handle list items (both - and numbered)
      else if (line.startsWith('- ') || /^\d+\.\s/.test(line)) {
        if (!inList) { 
          html += line.startsWith('- ') ? '<ul>' : '<ol>'; 
          inList = line.startsWith('- ') ? 'ul' : 'ol'; 
        }
        const content = line.startsWith('- ') ? line.substring(2) : line.replace(/^\d+\.\s/, '')
        html += `<li>${content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</li>`
      }
             // Handle regular text
       else if (line.trim()) {
         if (inList) { html += `</${inList}>`; inList = false; }
         html += `<p>${line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</p>`
       }
       // Handle empty lines
       else {
         if (inList) { html += `</${inList}>`; inList = false; }
         html += '<br>'
       }
     }
     
     // Close any open list
     if (inList) {
       html += `</${inList}>`
     }

    return html
  }

  if (loading) {
    return (
      <div className="docs-loading">
        <div className="loading-spinner"></div>
        <p>Loading documentation...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="docs-error">
        <h3>Error Loading Documentation</h3>
        <p>{error}</p>
      </div>
    )
  }

  const currentDomain = domainDocs[domain]
  const domainTools = toolsData ? toolsData[domain] : null

  return (
    <div className="docs-content-wrapper">
      <div className="docs-header">
        <h1>{currentDomain.title}</h1>
        <p className="docs-description">{currentDomain.description}</p>
      </div>

      <div className="docs-body">
        <div 
          className="policy-content"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(policyContent) }}
        />
        
        {domainTools && (
          <div className="tools-section">
            <h2>Available Tools & Functions</h2>
            <p className="tools-description">{domainTools.description}</p>
            
            <div className="tools-grid">
              {domainTools.tools.map((tool, index) => (
                <div key={index} className="tool-card">
                  <div className="tool-header">
                    <h3 className="tool-name">{tool.name}</h3>
                    <span className={`tool-type tool-type-${tool.type.toLowerCase()}`}>
                      {tool.type}
                    </span>
                  </div>
                  
                  <p className="tool-description">{tool.description}</p>
                  
                  {tool.parameters.length > 0 && (
                    <div className="tool-parameters">
                      <h4>Parameters</h4>
                      <div className="parameters-list">
                        {tool.parameters.map((param, paramIndex) => (
                          <div key={paramIndex} className="parameter-item">
                            <div className="parameter-header">
                              <span className="parameter-name">{param.name}</span>
                              <span className="parameter-type">{param.type}</span>
                              {param.required && <span className="parameter-required">required</span>}
                            </div>
                            <p className="parameter-description">{param.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="tool-returns">
                    <h4>Returns</h4>
                    <p>{tool.returns}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="docs-footer">
        <div className="docs-note">
          <h3>📖 About This Documentation</h3>
          <p>
            This documentation represents the agent policy and domain specifications for the {domain} domain in τ²-bench. 
            Agents are evaluated based on their adherence to these policies and their ability to successfully complete tasks within this domain.
          </p>
          <div className="docs-links">
            <a href="https://arxiv.org/abs/2506.07982" target="_blank" rel="noopener noreferrer">
              📄 Read the Full Paper
            </a>
            <a href="https://github.com/sierra-research/tau2-bench" target="_blank" rel="noopener noreferrer">
              💻 View on GitHub
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DocsContent 