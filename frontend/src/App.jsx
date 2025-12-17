import React, { useState } from 'react';
import { Search, TrendingUp, TrendingDown, Minus, Download, Loader2, AlertCircle, Info } from 'lucide-react';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Select from 'react-select';
import { ukStocks } from './ukStocks';
import { usStocks } from './usStocks';

const ShareAnalyzer = () => {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);

  // Email gate state
  const [analysisCount, setAnalysisCount] = useState(0);
  const [userEmail, setUserEmail] = useState(null);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [emailInput, setEmailInput] = useState('');
  const [emailSubmitting, setEmailSubmitting] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [showSuccessMessage, setShowSuccessMessage] = useState(false);

  // News filter state
  const [sentimentFilter, setSentimentFilter] = useState(null);
  const [categoryFilter, setCategoryFilter] = useState(null);

  // Filtered news based on selected filters
  const filteredNews = analysis?.news ? analysis.news.filter(article => {
    // Filter by sentiment
    if (sentimentFilter !== null && article.sentiment !== sentimentFilter) {
      return false;
    }
    
    // Filter by category
    if (categoryFilter !== null && article.category !== categoryFilter) {
      return false;
    }
    
    return true;
  }) : [];

  // Load saved user email from localStorage
  React.useEffect(() => {
    const savedEmail = localStorage.getItem('userEmail');
    const savedUserData = localStorage.getItem('userData');
    
    if (savedEmail) {
      setUserEmail(savedEmail);
    }
    
    if (savedUserData) {
      try {
        const userData = JSON.parse(savedUserData);
        console.log('Welcome back,', userData.firstName, userData.lastName);
      } catch (e) {
        console.error('Error parsing user data:', e);
      }
    }
  }, []);

  const analyzeStock = async () => {
    if (!ticker.trim()) return;
    
    // Check if user has reached limit and needs to provide email
    if (analysisCount >= 1 && !userEmail) {
      setShowEmailModal(true);
      return;
    }
    
    setLoading(true);
    setError(null);
    
  try{
    const response = await fetch('https://stock-analyzer-backend-83mw.onrender.com/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ ticker: ticker.toUpperCase() })
});
      
      const data = await response.json();
      
      if (data.success) {
        setAnalysis(data);
        setAnalysisCount(prev => prev + 1); // Increment analysis counter
      } else {
        setError('Failed to analyze stock');
      }
    } catch (err) {
      setError('Unable to connect to backend. Make sure the FastAPI server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const submitEmail = async () => {
    // Clear previous errors
    setValidationError('');
    
    // Validate First Name (mandatory)
    if (!firstName.trim()) {
      setValidationError('First Name is required');
      return;
    }
    
    // Validate First Name (at least 2 characters)
    if (firstName.trim().length < 2) {
      setValidationError('First Name must be at least 2 characters');
      return;
    }
    
    // Validate Email (mandatory)
    if (!emailInput.trim()) {
      setValidationError('Email is required');
      return;
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(emailInput)) {
      setValidationError('Please enter a valid email address');
      return;
    }
    
    setEmailSubmitting(true);
    
    try {
      const response = await fetch('https://stock-analyzer-backend-83mw.onrender.com/api/capture-email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          firstName: firstName.trim(),
          lastName: lastName.trim(),
          email: emailInput.trim().toLowerCase(),
          source: 'email_gate',
          timestamp: new Date().toISOString()
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Store user data in state and localStorage
        const userData = {
          firstName: firstName.trim(),
          lastName: lastName.trim(),
          email: emailInput.trim().toLowerCase()
        };
        
        setUserEmail(emailInput.trim().toLowerCase());
        localStorage.setItem('userEmail', emailInput.trim().toLowerCase());
        localStorage.setItem('userData', JSON.stringify(userData));
        
        setShowEmailModal(false);
        setShowSuccessMessage(true);  // ← ADD THIS
        
        // Hide success message after 7 seconds
        setTimeout(() => setShowSuccessMessage(false), 7000);  // ← ADD THIS
        
        // Clear form
        setFirstName('');
        setLastName('');
        setEmailInput('');
        setValidationError('');
        
        // DON'T call analyzeStock() here - removed to prevent modal reopening
      } else {

        setValidationError(data.message || 'Failed to save information. Please try again.');
      }
    } catch (err) {
      setValidationError('Error saving information. Please try again.');
    } finally {
      setEmailSubmitting(false);
    }
  };
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      analyzeStock();
    }
  };

  const generatePDF = () => {
    if (!analysis) return;
    
    try {
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.width;
      
      doc.setFontSize(24);
      doc.setTextColor(59, 130, 246);
      doc.text('Stock Analyzer Report', pageWidth / 2, 20, { align: 'center' });
      
      doc.setFontSize(16);
      doc.setTextColor(0, 0, 0);
      doc.text(analysis.company.name, pageWidth / 2, 35, { align: 'center' });
      
      doc.setFontSize(12);
      doc.setTextColor(100, 100, 100);
      doc.text(`Ticker: ${analysis.company.ticker}`, pageWidth / 2, 42, { align: 'center' });
      doc.text(`Price: ${analysis.company.currency} ${analysis.company.price}`, pageWidth / 2, 49, { align: 'center' });
      doc.text(`Date: ${new Date(analysis.timestamp).toLocaleDateString()}`, pageWidth / 2, 56, { align: 'center' });
      
      let recColor;
      if (analysis.recommendation.recommendation === 'BUY') {
        recColor = [34, 197, 94];
      } else if (analysis.recommendation.recommendation === 'SELL') {
        recColor = [239, 68, 68];
      } else {
        recColor = [234, 179, 8];
      }
      
      doc.setFillColor(...recColor);
      doc.rect(15, 65, pageWidth - 30, 25, 'F');
      
      doc.setFontSize(18);
      doc.setTextColor(255, 255, 255);
      doc.text(`RECOMMENDATION: ${analysis.recommendation.recommendation}`, pageWidth / 2, 75, { align: 'center' });
      doc.text(`Overall Score: ${analysis.recommendation.score}/100`, pageWidth / 2, 85, { align: 'center' });
      
      let yPosition = 100;
      
      doc.setFontSize(14);
      doc.setTextColor(59, 130, 246);
      doc.text('Valuation Metrics', 15, yPosition);
      yPosition += 5;
      
      autoTable(doc, {
        startY: yPosition,
        head: [['Metric', 'Value', 'Score']],
        body: [
          ['P/E Ratio', analysis.kpis.valuation.pe_ratio.value || 'N/A', `${analysis.kpis.valuation.pe_ratio.score}/10`],
          ['P/B Ratio', analysis.kpis.valuation.pb_ratio.value || 'N/A', `${analysis.kpis.valuation.pb_ratio.score}/10`],
        ],
        theme: 'grid',
        headStyles: { fillColor: [59, 130, 246] },
      });
      
      yPosition = doc.lastAutoTable.finalY + 10;
      
      doc.setTextColor(34, 197, 94);
      doc.text('Profitability Metrics', 15, yPosition);
      yPosition += 5;
      
      autoTable(doc, {
        startY: yPosition,
        head: [['Metric', 'Value', 'Score']],
        body: [
          ['ROE', analysis.kpis.profitability.roe.value ? `${analysis.kpis.profitability.roe.value}%` : 'N/A', `${analysis.kpis.profitability.roe.score}/10`],
          ['Profit Margin', analysis.kpis.profitability.profit_margin.value ? `${analysis.kpis.profitability.profit_margin.value}%` : 'N/A', `${analysis.kpis.profitability.profit_margin.score}/10`],
          ['Operating Margin', analysis.kpis.profitability.operating_margin.value ? `${analysis.kpis.profitability.operating_margin.value}%` : 'N/A', `${analysis.kpis.profitability.operating_margin.score}/10`],
        ],
        theme: 'grid',
        headStyles: { fillColor: [34, 197, 94] },
      });
      
      yPosition = doc.lastAutoTable.finalY + 10;
      
      doc.setTextColor(168, 85, 247);
      doc.text('Financial Health', 15, yPosition);
      yPosition += 5;
      
      autoTable(doc, {
        startY: yPosition,
        head: [['Metric', 'Value', 'Score']],
        body: [
          ['Debt-to-Equity', analysis.kpis.health.debt_to_equity.value || 'N/A', `${analysis.kpis.health.debt_to_equity.score}/10`],
          ['Current Ratio', analysis.kpis.health.current_ratio.value || 'N/A', `${analysis.kpis.health.current_ratio.score}/10`],
        ],
        theme: 'grid',
        headStyles: { fillColor: [168, 85, 247] },
      });
      
      doc.addPage();
      yPosition = 20;
      
      doc.setTextColor(234, 179, 8);
      doc.text('Growth Metrics', 15, yPosition);
      yPosition += 5;
      
      autoTable(doc, {
        startY: yPosition,
        head: [['Metric', 'Value', 'Score']],
        body: [
          ['Revenue Growth', analysis.kpis.growth.revenue_growth.value ? `${analysis.kpis.growth.revenue_growth.value}%` : 'N/A', `${analysis.kpis.growth.revenue_growth.score}/10`],
          ['EPS Growth', analysis.kpis.growth.eps_growth.value ? `${analysis.kpis.growth.eps_growth.value}%` : 'N/A', `${analysis.kpis.growth.eps_growth.score}/10`],
        ],
        theme: 'grid',
        headStyles: { fillColor: [234, 179, 8] },
      });
      
      yPosition = doc.lastAutoTable.finalY + 10;
      
      doc.setTextColor(249, 115, 22);
      doc.text('Technical & Market', 15, yPosition);
      yPosition += 5;
      
      autoTable(doc, {
        startY: yPosition,
        head: [['Metric', 'Value', 'Score']],
        body: [
          ['Beta', analysis.kpis.technical.beta.value || 'N/A', `${analysis.kpis.technical.beta.score}/10`],
          ['52W Price Position', analysis.kpis.technical.price_position.value ? `${analysis.kpis.technical.price_position.value}%` : 'N/A', `${analysis.kpis.technical.price_position.score}/10`],
          ['Dividend Yield', analysis.kpis.technical.dividend_yield.value ? `${analysis.kpis.technical.dividend_yield.value}%` : 'N/A', `${analysis.kpis.technical.dividend_yield.score}/10`],
        ],
        theme: 'grid',
        headStyles: { fillColor: [249, 115, 22] },
      });
      
      yPosition = doc.lastAutoTable.finalY + 15;
      
      doc.setFontSize(10);
      doc.setTextColor(100, 100, 100);
      const disclaimer = 'Disclaimer: This analysis is for informational purposes only and does not constitute financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.';
      const splitDisclaimer = doc.splitTextToSize(disclaimer, pageWidth - 30);
      doc.text(splitDisclaimer, 15, yPosition);
      
      const fileName = `${analysis.company.ticker}_Analysis_${new Date().toISOString().split('T')[0]}.pdf`;
      doc.save(fileName);
      
      console.log('PDF generated successfully:', fileName);
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Error generating PDF: ' + error.message);
    }
  };

  const getRecommendationIcon = (rec) => {
    if (rec === 'BUY') return <TrendingUp className="w-8 h-8" />;
    if (rec === 'SELL') return <TrendingDown className="w-8 h-8" />;
    return <Minus className="w-8 h-8" />;
  };

  const getRecommendationColor = (rec) => {
    if (rec === 'BUY') return 'from-green-500 to-emerald-600';
    if (rec === 'SELL') return 'from-red-500 to-rose-600';
    return 'from-yellow-500 to-amber-600';
  };

  const ScoreBar = ({ score }) => {
    const percentage = (score / 10) * 100;
    const getColor = () => {
      if (score >= 7) return 'bg-green-500';
      if (score >= 4) return 'bg-yellow-500';
      return 'bg-red-500';
    };
    
    return (
      <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
        <div 
          className={`h-2 rounded-full ${getColor()} transition-all duration-500`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    );
  };

  const KPICard = ({ label, value, unit, score, tooltip }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    
    return (
      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center gap-2 mb-1">
          <div className="text-gray-400 text-sm">{label}</div>
          {tooltip && (
            <div className="relative">
              <Info 
                className="w-4 h-4 text-gray-500 cursor-help hover:text-gray-300 transition-colors"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
              />
              {showTooltip && (
                <div className="absolute z-10 w-64 p-3 bg-gray-900 border border-gray-600 rounded-lg shadow-xl text-xs text-gray-300 -top-2 left-6">
                  <div className="font-semibold text-white mb-1">{label}</div>
                  {tooltip}
                </div>
              )}
            </div>
          )}
        </div>
        <div className="text-2xl font-bold text-white mb-2">
          {value !== null ? `${value}${unit}` : 'N/A'}
        </div>
        <ScoreBar score={score} />
        <div className="text-xs text-gray-500 mt-1">Score: {score}/10</div>
      </div>
    );
  };

  // KPI Tooltips with explanations and formulas
  const kpiTooltips = {
    'P/E Ratio': (
      <div>
        <p className="mb-1">Price-to-Earnings ratio measures how much investors are willing to pay per pound of earnings.</p>
        <p className="font-mono text-gray-400">Formula: Stock Price ÷ Earnings Per Share</p>
        <p className="mt-1 text-gray-400">Lower values may indicate undervaluation.</p>
      </div>
    ),
    'P/B Ratio': (
      <div>
        <p className="mb-1">Price-to-Book ratio compares market value to book value of assets.</p>
        <p className="font-mono text-gray-400">Formula: Stock Price ÷ Book Value Per Share</p>
        <p className="mt-1 text-gray-400">Values under 1.0 may indicate undervaluation.</p>
      </div>
    ),
    'ROE': (
      <div>
        <p className="mb-1">Return on Equity measures profitability relative to shareholders' equity.</p>
        <p className="font-mono text-gray-400">Formula: (Net Income ÷ Shareholders' Equity) × 100</p>
        <p className="mt-1 text-gray-400">Higher values indicate better returns for shareholders.</p>
      </div>
    ),
    'Profit Margin': (
      <div>
        <p className="mb-1">Profit Margin shows what percentage of revenue becomes profit.</p>
        <p className="font-mono text-gray-400">Formula: (Net Income ÷ Revenue) × 100</p>
        <p className="mt-1 text-gray-400">Higher margins indicate better profitability.</p>
      </div>
    ),
    'Operating Margin': (
      <div>
        <p className="mb-1">Operating Margin measures operational efficiency before interest and taxes.</p>
        <p className="font-mono text-gray-400">Formula: (Operating Income ÷ Revenue) × 100</p>
        <p className="mt-1 text-gray-400">Shows core business profitability.</p>
      </div>
    ),
    'Debt-to-Equity': (
      <div>
        <p className="mb-1">Debt-to-Equity ratio measures financial leverage and risk.</p>
        <p className="font-mono text-gray-400">Formula: Total Debt ÷ Total Equity</p>
        <p className="mt-1 text-gray-400">Lower values indicate less financial risk.</p>
      </div>
    ),
    'Current Ratio': (
      <div>
        <p className="mb-1">Current Ratio measures ability to pay short-term obligations.</p>
        <p className="font-mono text-gray-400">Formula: Current Assets ÷ Current Liabilities</p>
        <p className="mt-1 text-gray-400">Values above 1.0 indicate good liquidity.</p>
      </div>
    ),
    'Revenue Growth': (
      <div>
        <p className="mb-1">Revenue Growth shows year-over-year increase in total sales.</p>
        <p className="font-mono text-gray-400">Formula: ((Current Revenue - Prior Revenue) ÷ Prior Revenue) × 100</p>
        <p className="mt-1 text-gray-400">Positive growth indicates business expansion.</p>
      </div>
    ),
    'EPS Growth': (
      <div>
        <p className="mb-1">Earnings Per Share Growth measures year-over-year profit increase per share.</p>
        <p className="font-mono text-gray-400">Formula: ((Current EPS - Prior EPS) ÷ Prior EPS) × 100</p>
        <p className="mt-1 text-gray-400">Higher growth indicates increasing profitability.</p>
      </div>
    ),
    'Beta': (
      <div>
        <p className="mb-1">Beta measures stock volatility relative to the overall market.</p>
        <p className="font-mono text-gray-400">Beta = 1.0 means moves with market</p>
        <p className="font-mono text-gray-400">Beta &gt; 1.0 means more volatile</p>
        <p className="font-mono text-gray-400">Beta &lt; 1.0 means less volatile</p>
      </div>
    ),
    '52W Price Position': (
      <div>
        <p className="mb-1">Shows current price position within the 52-week trading range.</p>
        <p className="font-mono text-gray-400">Formula: ((Current - 52W Low) ÷ (52W High - 52W Low)) × 100</p>
        <p className="mt-1 text-gray-400">0% = at 52-week low, 100% = at 52-week high</p>
      </div>
    ),
    'Dividend Yield': (
      <div>
        <p className="mb-1">Dividend Yield shows annual dividend income as percentage of stock price.</p>
        <p className="font-mono text-gray-400">Formula: (Annual Dividends Per Share ÷ Stock Price) × 100</p>
        <p className="mt-1 text-gray-400">Higher yields provide more income.</p>
      </div>
    ),
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
 
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <TrendingUp className="w-12 h-12 text-blue-400" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
              Stock Analyzer
            </h1>
          </div>
          <p className="text-xl text-gray-300">
            Comprehensive analysis of UK & US stocks • AI-powered insights • Market data
          </p>
          <p className="text-sm text-gray-400 mt-2">
            Analyze 100+ UK stocks (FTSE 100/250) and 100 US stocks (S&P 500 & major indices)
          </p>
        </div>

        {/* Search Section */}
        <div className="mb-12">
          <div className="flex gap-4 max-w-2xl mx-auto">
            <div className="flex-1 relative">
              <Select
                options={[
                  {
                    label: '🇬🇧 UK Stocks',
                    options: ukStocks.map(stock => ({
                      value: stock.symbol,
                      label: `${stock.name} (${stock.symbol})`,
                      market: 'UK'
                    }))
                  },
                  {
                    label: '🇺🇸 US Stocks',
                    options: usStocks.map(stock => ({
                      value: stock.symbol,
                      label: `${stock.name} (${stock.symbol})`,
                      market: 'US'
                    }))
                  }
                ]}
                onChange={(selected) => {
                  if (selected) {
                    setTicker(selected.value);
                  } else {
                    setTicker('');
                  }
                }}
                placeholder="Search stocks by name or ticker (e.g., Apple, AAPL, Barclays, BARC.L)..."
                isClearable
                isSearchable
                noOptionsMessage={() => "No stocks found"}
                styles={{
                  control: (base, state) => ({
                    ...base,
                    backgroundColor: '#1f2937',
                    borderColor: state.isFocused ? '#3b82f6' : '#374151',
                    padding: '8px',
                    borderRadius: '8px',
                    boxShadow: state.isFocused ? '0 0 0 1px #3b82f6' : 'none',
                    '&:hover': {
                      borderColor: '#3b82f6'
                    }
                  }),
                  input: (base) => ({
                    ...base,
                    color: '#fff'
                  }),
                  placeholder: (base) => ({
                    ...base,
                    color: '#9ca3af'
                  }),
                  singleValue: (base) => ({
                    ...base,
                    color: '#fff'
                  }),
                  menu: (base) => ({
                    ...base,
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    marginTop: '4px',
                    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3)'
                  }),
                  option: (base, state) => ({
                    ...base,
                    backgroundColor: state.isFocused ? '#374151' : '#1f2937',
                    color: '#fff',
                    padding: '12px 16px',
                    cursor: 'pointer',
                    '&:active': {
                      backgroundColor: '#4b5563'
                    }
                  }),
                  menuList: (base) => ({
                    ...base,
                    maxHeight: '400px',
                    padding: '4px'
                  }),
                  group: (base) => ({
                    ...base,
                    paddingTop: '8px',
                    paddingBottom: '8px'
                  }),
                  groupHeading: (base) => ({
                    ...base,
                    fontSize: '14px',
                    fontWeight: 'bold',
                    color: '#3b82f6',
                    textTransform: 'none',
                    padding: '8px 16px',
                    marginBottom: '4px',
                    borderBottom: '1px solid #374151'
                  }),
                  clearIndicator: (base) => ({
                    ...base,
                    color: '#9ca3af',
                    cursor: 'pointer',
                    '&:hover': {
                      color: '#fff'
                    }
                  }),
                  dropdownIndicator: (base) => ({
                    ...base,
                    color: '#9ca3af',
                    cursor: 'pointer',
                    '&:hover': {
                      color: '#fff'
                    }
                  })
                }}
                formatOptionLabel={(option) => (
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col flex-1">
                      <div className="font-semibold">{option.label}</div>
                    </div>
                    {option.market && (
                      <span className={`ml-2 px-2 py-0.5 rounded text-xs font-medium ${
                        option.market === 'UK' 
                          ? 'bg-blue-500/20 text-blue-300' 
                          : 'bg-purple-500/20 text-purple-300'
                      }`}>
                        {option.market}
                      </span>
                    )}
                  </div>
                )}
              />
            </div>
            <button
              onClick={analyzeStock}
              disabled={loading || !ticker.trim()}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg font-semibold hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing
                </>
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  Analyze
                </>
              )}
            </button>
          </div>
          
          {error && (
            <div className="mt-4 max-w-2xl mx-auto p-4 bg-red-900/20 border border-red-500/50 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-300">{error}</p>
            </div>
          )}
        </div>

        {/* Analysis Results */}
        {analysis && (
          <div className="space-y-8 animate-fadeIn">
            {/* Company Header */}
            <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-3xl font-bold mb-2">{analysis.company.name}</h2>
                  <p className="text-gray-400 text-lg">{analysis.company.ticker}</p>
                </div>
                <div className="text-right">
                  <div className="text-3xl font-bold">
                    {analysis.company.currency} {analysis.company.price}
                  </div>
                </div>
              </div>
            </div>

            {/* Recommendation Card */}
            <div className={`bg-gradient-to-r ${getRecommendationColor(analysis.recommendation.recommendation)} rounded-xl p-8 shadow-2xl`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-6">
                  {getRecommendationIcon(analysis.recommendation.recommendation)}
                  <div>
                    <div className="text-sm opacity-90 mb-1">RECOMMENDATION</div>
                    <div className="text-4xl font-bold">{analysis.recommendation.recommendation}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm opacity-90 mb-1">OVERALL SCORE</div>
                  <div className="text-5xl font-bold">{analysis.recommendation.score}/100</div>
                </div>
              </div>
              <p className="mt-4 text-lg opacity-90">{analysis.recommendation.message}</p>
            </div>

            {/* 12-Month Price Chart WITH MOVING AVERAGES */}
            {analysis.historical_prices && analysis.historical_prices.length > 0 && (
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
                <h3 className="text-2xl font-bold text-blue-400 mb-6">12-Month Price Chart with Moving Averages</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={analysis.historical_prices}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="date" 
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                      tickFormatter={(date) => {
                        const d = new Date(date);
                        return `${d.getMonth() + 1}/${d.getFullYear().toString().slice(-2)}`;
                      }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                      domain={['auto', 'auto']}
                      tickFormatter={(value) => `${analysis.company.currency} ${value.toFixed(2)}`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#fff'
                      }}
                      labelFormatter={(date) => new Date(date).toLocaleDateString()}
                      formatter={(value, name) => {
                        const labels = {
                          'price': 'Price',
                          'ma_50': '50-Day MA',
                          'ma_200': '200-Day MA'
                        };
                        return [`${analysis.company.currency} ${value.toFixed(2)}`, labels[name] || name];
                      }}
                    />
                    
                    {/* Main Price Line */}
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#3B82F6" 
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 6 }}
                      name="Price"
                    />
                    
                    {/* 50-Day Moving Average */}
                    <Line 
                      type="monotone" 
                      dataKey="ma_50" 
                      stroke="#F59E0B" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      name="50-Day MA"
                    />
                    
                    {/* 200-Day Moving Average */}
                    <Line 
                      type="monotone" 
                      dataKey="ma_200" 
                      stroke="#EF4444" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                      name="200-Day MA"
                    />
                  </LineChart>
                </ResponsiveContainer>
                
                {/* Legend */}
                <div className="mt-4 flex items-center justify-center gap-6 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-0.5" style={{backgroundColor: '#3B82F6'}}></div>
                    <span className="text-gray-300">Price</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-8" style={{height: '2px', borderTop: '2px dashed #F59E0B'}}></div>
                    <span className="text-gray-300">50-Day MA</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-8" style={{height: '2px', borderTop: '2px dashed #EF4444'}}></div>
                    <span className="text-gray-300">200-Day MA</span>
                  </div>
                </div>
                
                <div className="mt-4 text-sm text-gray-400 text-center">
                  Historical closing prices with 50-day and 200-day moving averages
                </div>
              </div>
            )}

            {/* Company Overview */}
            {analysis.company_overview && (
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-bold text-purple-400">Company Overview</h3>
                  
                  {/* Exchange and Index Badges */}
                  <div className="flex gap-2">
                    {analysis.company_overview.exchange && (
                      <span className="px-3 py-1 bg-blue-500/20 text-blue-300 text-sm font-medium rounded-full border border-blue-500/30">
                        {analysis.company_overview.exchange_full || analysis.company_overview.exchange}
                      </span>
                    )}
                    {analysis.company_overview.index && (
                      <span className="px-3 py-1 bg-green-500/20 text-green-300 text-sm font-medium rounded-full border border-green-500/30">
                        {analysis.company_overview.index}
                      </span>
                    )}
                  </div>
                </div>
                
                {/* Company Description */}
                {analysis.company_overview.description && (
                  <div className="mb-6">
                    <p className="text-gray-300 leading-relaxed">
                      {analysis.company_overview.description}
                    </p>
                  </div>
                )}
                
                {/* Company Details Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  
                  {/* Sector & Industry */}
                  {analysis.company_overview.sector && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Sector</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company_overview.sector}
                      </div>
                    </div>
                  )}
                  
                  {analysis.company_overview.industry && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Industry</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company_overview.industry}
                      </div>
                    </div>
                  )}
                  
                  {/* Employees */}
                  {analysis.company_overview.employees && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Employees</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company_overview.employees.toLocaleString()}
                      </div>
                    </div>
                  )}
                  
                  {/* Headquarters */}
                  {analysis.company_overview.headquarters && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Headquarters</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company_overview.headquarters}
                      </div>
                    </div>
                  )}
                  
                  {/* Market Cap */}
                  {analysis.company_overview.market_cap && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Market Cap</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company.currency.replace('GBp', 'GBP').replace('GBX', 'GBP')} {(analysis.company_overview.market_cap / 1e9).toFixed(2)}B
                      </div>
                    </div>
                  )}
                  
                  {/* Enterprise Value */}
                  {analysis.company_overview.enterprise_value && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Enterprise Value</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company.currency.replace('GBp', 'GBP').replace('GBX', 'GBP')} {(analysis.company_overview.enterprise_value / 1e9).toFixed(2)}B
                      </div>
                    </div>
                  )}
                  
                  {/* Day Range */}
                  {analysis.company_overview.day_range && analysis.company_overview.day_range.low && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Day Range</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company.currency.replace('GBp', 'GBP').replace('GBX', 'GBP')} {analysis.company_overview.day_range.low.toFixed(2)} - {analysis.company_overview.day_range.high.toFixed(2)}
                      </div>
                    </div>
                  )}
                  
                  {/* 52 Week Range */}
                  {analysis.company_overview.week_52_range && analysis.company_overview.week_52_range.low && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">52 Week Range</div>
                      <div className="text-lg font-semibold text-white">
                        {analysis.company.currency.replace('GBp', 'GBP').replace('GBX', 'GBP')} {analysis.company_overview.week_52_range.low.toFixed(2)} - {analysis.company_overview.week_52_range.high.toFixed(2)}
                      </div>
                    </div>
                  )}
                  
                  {/* Volume */}
                  {analysis.company_overview.volume && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Volume</div>
                      <div className="text-lg font-semibold text-white">
                        {(analysis.company_overview.volume / 1e6).toFixed(2)}M
                      </div>
                    </div>
                  )}
                  
                  {/* Average Volume */}
                  {analysis.company_overview.avg_volume && (
                    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
                      <div className="text-sm text-gray-400 mb-1">Avg Volume</div>
                      <div className="text-lg font-semibold text-white">
                        {(analysis.company_overview.avg_volume / 1e6).toFixed(2)}M
                      </div>
                    </div>
                  )}
                  
                </div>
                
                {/* Website Link */}
                {analysis.company_overview.website && (
                  <div className="mt-6">
                    <a 
                      href={analysis.company_overview.website} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors"
                    >
                      <span>Visit Company Website</span>
                      <span>→</span>
                    </a>
                  </div>
                )}
              </div>
            )}

            {/* Latest News with Filters */}
            {analysis.news && analysis.news.length > 0 && (
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
                <h3 className="text-2xl font-bold text-blue-400 mb-4">Latest News</h3>
                
                {/* Filters Section */}
                <div className="mb-6 space-y-3">
                  {/* Sentiment Filters */}
                  <div>
                    <p className="text-sm text-gray-400 mb-2">Filter by Sentiment:</p>
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => setSentimentFilter(null)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          sentimentFilter === null
                            ? 'bg-blue-500 text-white border border-blue-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        All
                      </button>
                      <button
                        onClick={() => setSentimentFilter('positive')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          sentimentFilter === 'positive'
                            ? 'bg-green-500/30 text-green-200 border border-green-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        🟢 Positive
                      </button>
                      <button
                        onClick={() => setSentimentFilter('negative')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          sentimentFilter === 'negative'
                            ? 'bg-red-500/30 text-red-200 border border-red-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        🔴 Negative
                      </button>
                      <button
                        onClick={() => setSentimentFilter('neutral')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          sentimentFilter === 'neutral'
                            ? 'bg-gray-500/30 text-gray-200 border border-gray-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        ⚪ Neutral
                      </button>
                    </div>
                  </div>

                  {/* Category Filters */}
                  <div>
                    <p className="text-sm text-gray-400 mb-2">Filter by Category:</p>
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => setCategoryFilter(null)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          categoryFilter === null
                            ? 'bg-blue-500 text-white border border-blue-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        All
                      </button>
                      <button
                        onClick={() => setCategoryFilter('earnings')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          categoryFilter === 'earnings'
                            ? 'bg-purple-500/30 text-purple-200 border border-purple-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        💰 Earnings
                      </button>
                      <button
                        onClick={() => setCategoryFilter('ma')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          categoryFilter === 'ma'
                            ? 'bg-orange-500/30 text-orange-200 border border-orange-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        🤝 M&A
                      </button>
                      <button
                        onClick={() => setCategoryFilter('regulation')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          categoryFilter === 'regulation'
                            ? 'bg-red-500/30 text-red-200 border border-red-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        ⚖️ Regulation
                      </button>
                      <button
                        onClick={() => setCategoryFilter('markets')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          categoryFilter === 'markets'
                            ? 'bg-sky-500/30 text-sky-200 border border-sky-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        📊 Markets
                      </button>
                      <button
                        onClick={() => setCategoryFilter('leadership')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          categoryFilter === 'leadership'
                            ? 'bg-emerald-500/30 text-emerald-200 border border-emerald-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        👔 Leadership
                      </button>
                      <button
                        onClick={() => setCategoryFilter('general')}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          categoryFilter === 'general'
                            ? 'bg-gray-500/30 text-gray-200 border border-gray-400'
                            : 'bg-gray-700/50 text-gray-300 border border-gray-600 hover:bg-gray-700'
                        }`}
                      >
                        📰 General
                      </button>
                    </div>
                  </div>

                  {/* Filter Status and Clear */}
                  <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                    <p className="text-sm text-gray-400">
                      Showing <span className="text-blue-300 font-semibold">{filteredNews.length}</span> of {analysis.news.length} articles
                    </p>
                    {(sentimentFilter !== null || categoryFilter !== null) && (
                      <button
                        onClick={() => {
                          setSentimentFilter(null);
                          setCategoryFilter(null);
                        }}
                        className="px-3 py-1 rounded text-sm text-blue-300 hover:text-blue-200 hover:bg-blue-500/10 transition-all"
                      >
                        ✕ Clear Filters
                      </button>
                    )}
                  </div>
                </div>

                {/* News Articles */}
                <div className="space-y-4">
                  {filteredNews.length > 0 ? (
                    filteredNews.map((article, idx) => {
                      // Helper function to get category display info
                      const getCategoryInfo = (category) => {
                        const categories = {
                          'earnings': { icon: '💰', label: 'Earnings', color: 'purple' },
                          'ma': { icon: '🤝', label: 'M&A', color: 'orange' },
                          'regulation': { icon: '⚖️', label: 'Regulation', color: 'red' },
                          'markets': { icon: '📊', label: 'Markets', color: 'blue' },
                          'leadership': { icon: '👔', label: 'Leadership', color: 'green' },
                          'general': { icon: '📰', label: 'General', color: 'gray' }
                        };
                        return categories[category] || categories['general'];
                      };

                      const categoryInfo = article.category ? getCategoryInfo(article.category) : null;

                      return (
                        <a
                          key={idx}
                          href={article.link || '#'}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block p-4 bg-gray-700/50 rounded-lg border border-gray-600 hover:bg-gray-700/70 hover:border-blue-500/50 transition-all"
                        >
                          <div className="flex gap-4">
                            {article.thumbnail && (
                              <img
                                src={article.thumbnail}
                                alt="article"
                                className="w-32 h-24 object-cover rounded-lg flex-shrink-0"
                                onError={(e) => e.target.style.display = 'none'}
                              />
                            )}
                            <div className="flex-1 min-w-0">
                              {/* Badge Row: Sentiment, Read Time, Category */}
                              <div className="flex flex-wrap items-center gap-2 mb-2">
                                {article.sentiment && (
                                  <span className={`flex-shrink-0 px-2 py-0.5 rounded text-xs font-medium ${
                                    article.sentiment === 'positive' 
                                      ? 'bg-green-500/20 text-green-300 border border-green-500/30' 
                                      : article.sentiment === 'negative'
                                      ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                                      : 'bg-gray-500/20 text-gray-300 border border-gray-500/30'
                                  }`}>
                                    {article.sentiment === 'positive' ? '🟢 Positive' : 
                                     article.sentiment === 'negative' ? '🔴 Negative' : 
                                     '⚪ Neutral'}
                                  </span>
                                )}
                                {article.read_time && (
                                  <span className="flex-shrink-0 px-2 py-0.5 rounded text-xs font-medium bg-blue-500/20 text-blue-300 border border-blue-500/30">
                                    📖 {article.read_time} min
                                  </span>
                                )}
                                {categoryInfo && (
                                  <span className={`flex-shrink-0 px-2 py-0.5 rounded text-xs font-medium ${
                                    categoryInfo.color === 'purple' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' :
                                    categoryInfo.color === 'orange' ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30' :
                                    categoryInfo.color === 'red' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                                    categoryInfo.color === 'blue' ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30' :
                                    categoryInfo.color === 'green' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
                                    'bg-gray-500/20 text-gray-300 border border-gray-500/30'
                                  }`}>
                                    {categoryInfo.icon} {categoryInfo.label}
                                  </span>
                                )}
                              </div>
                              
                              <h4 className="text-lg font-semibold text-white hover:text-blue-300 transition-colors mb-2">
                                {article.title || 'No title'}
                              </h4>
                              
                              {article.summary && (
                                <p className="text-sm text-gray-300 mb-2 line-clamp-2">
                                  {article.summary}
                                </p>
                              )}
                              <div className="flex gap-4 text-sm text-gray-400">
                                <span>{article.publisher || 'Unknown'}</span>
                                <span>
                                  {article.publish_time > 0 ? new Date(article.publish_time * 1000).toLocaleDateString() : 'Recent'}
                                </span>
                              </div>
                            </div>
                          </div>
                        </a>
                      );
                    })
                  ) : (
                    <div className="text-center py-12">
                      <p className="text-gray-400 text-lg mb-2">No articles match your filters</p>
                      <button
                        onClick={() => {
                          setSentimentFilter(null);
                          setCategoryFilter(null);
                        }}
                        className="text-blue-300 hover:text-blue-200 text-sm"
                      >
                        Clear filters to see all articles
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* KPI Sections */}
            <div className="space-y-6">
              {/* Valuation */}
              <div className="bg-gray-800/30 rounded-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-blue-400">Valuation Metrics</h3>
                  <div className="text-lg">
                    Score: <span className="font-bold">{analysis.recommendation.category_scores.valuation}/10</span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <KPICard 
                    label="P/E Ratio" 
                    value={analysis.kpis.valuation.pe_ratio.value} 
                    unit="" 
                    score={analysis.kpis.valuation.pe_ratio.score}
                    tooltip={kpiTooltips['P/E Ratio']}
                  />
                  <KPICard 
                    label="P/B Ratio" 
                    value={analysis.kpis.valuation.pb_ratio.value} 
                    unit="" 
                    score={analysis.kpis.valuation.pb_ratio.score}
                    tooltip={kpiTooltips['P/B Ratio']}
                  />
                </div>
              </div>

              {/* Profitability */}
              <div className="bg-gray-800/30 rounded-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-green-400">Profitability Metrics</h3>
                  <div className="text-lg">
                    Score: <span className="font-bold">{analysis.recommendation.category_scores.profitability}/10</span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <KPICard 
                    label="ROE" 
                    value={analysis.kpis.profitability.roe.value} 
                    unit="%" 
                    score={analysis.kpis.profitability.roe.score}
                    tooltip={kpiTooltips['ROE']}
                  />
                  <KPICard 
                    label="Profit Margin" 
                    value={analysis.kpis.profitability.profit_margin.value} 
                    unit="%" 
                    score={analysis.kpis.profitability.profit_margin.score}
                    tooltip={kpiTooltips['Profit Margin']}
                  />
                  <KPICard 
                    label="Operating Margin" 
                    value={analysis.kpis.profitability.operating_margin.value} 
                    unit="%" 
                    score={analysis.kpis.profitability.operating_margin.score}
                    tooltip={kpiTooltips['Operating Margin']}
                  />
                </div>
              </div>

              {/* Financial Health */}
              <div className="bg-gray-800/30 rounded-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-purple-400">Financial Health</h3>
                  <div className="text-lg">
                    Score: <span className="font-bold">{analysis.recommendation.category_scores.health}/10</span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <KPICard 
                    label="Debt-to-Equity" 
                    value={analysis.kpis.health.debt_to_equity.value} 
                    unit="" 
                    score={analysis.kpis.health.debt_to_equity.score}
                    tooltip={kpiTooltips['Debt-to-Equity']}
                  />
                  <KPICard 
                    label="Current Ratio" 
                    value={analysis.kpis.health.current_ratio.value} 
                    unit="" 
                    score={analysis.kpis.health.current_ratio.score}
                    tooltip={kpiTooltips['Current Ratio']}
                  />
                </div>
              </div>

              {/* Growth */}
              <div className="bg-gray-800/30 rounded-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-yellow-400">Growth Metrics</h3>
                  <div className="text-lg">
                    Score: <span className="font-bold">{analysis.recommendation.category_scores.growth}/10</span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <KPICard 
                    label="Revenue Growth" 
                    value={analysis.kpis.growth.revenue_growth.value} 
                    unit="%" 
                    score={analysis.kpis.growth.revenue_growth.score}
                    tooltip={kpiTooltips['Revenue Growth']}
                  />
                  <KPICard 
                    label="EPS Growth" 
                    value={analysis.kpis.growth.eps_growth.value} 
                    unit="%" 
                    score={analysis.kpis.growth.eps_growth.score}
                    tooltip={kpiTooltips['EPS Growth']}
                  />
                </div>
              </div>

              {/* Technical */}
              <div className="bg-gray-800/30 rounded-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-2xl font-bold text-orange-400">Technical & Market</h3>
                  <div className="text-lg">
                    Score: <span className="font-bold">{analysis.recommendation.category_scores.technical}/10</span>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <KPICard 
                    label="Beta" 
                    value={analysis.kpis.technical.beta.value} 
                    unit="" 
                    score={analysis.kpis.technical.beta.score}
                    tooltip={kpiTooltips['Beta']}
                  />
                  <KPICard 
                    label="52W Price Position" 
                    value={analysis.kpis.technical.price_position.value} 
                    unit="%" 
                    score={analysis.kpis.technical.price_position.score}
                    tooltip={kpiTooltips['52W Price Position']}
                  />
                  <KPICard 
                    label="Dividend Yield" 
                    value={analysis.kpis.technical.dividend_yield.value} 
                    unit="%" 
                    score={analysis.kpis.technical.dividend_yield.score}
                    tooltip={kpiTooltips['Dividend Yield']}
                  />
                </div>
              </div>
            </div>

            {/* PDF Download Button */}
            <div className="flex justify-center pt-8">
              <button
                onClick={generatePDF}
                className="px-8 py-4 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg font-semibold hover:from-indigo-600 hover:to-purple-700 transition-all flex items-center gap-3 text-lg"
              >
                <Download className="w-6 h-6" />
                Download PDF Report
              </button>
            </div>

            {/* Disclaimer */}
            <div className="mt-8 p-4 bg-yellow-900/20 border border-yellow-700/50 rounded-lg">
              <p className="text-sm text-yellow-200">
                <strong>Disclaimer:</strong> This analysis is for informational purposes only and does not constitute financial advice. 
                Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
              </p>
            </div>
          </div>
        )}
        {/* Email Gate Modal */}
        {showEmailModal && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-2xl p-8 max-w-md w-full border-2 border-blue-500/50 shadow-2xl animate-fadeIn">
              <div className="text-center mb-6">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <TrendingUp className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-2">
                  Unlock Unlimited Analysis
                </h3>
                <p className="text-gray-300">
                  Get free access to analyze unlimited stocks, news, charts, and KPIs. 
                  Plus, receive weekly market insights in your inbox!
                </p>
              </div>
              
              <div className="space-y-4">
                {/* First Name - MANDATORY */}
                <div>
                  <input
                    type="text"
                    placeholder="First Name"
                    value={firstName}
                    onChange={(e) => setFirstName(e.target.value)}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                {/* Last Name - OPTIONAL */}
                <div>
                  <input
                    type="text"
                    placeholder="Last Name"
                    value={lastName}
                    onChange={(e) => setLastName(e.target.value)}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                {/* Email - MANDATORY */}
                <div>
                  <input
                    type="email"
                    placeholder="Enter Best Email"
                    value={emailInput}
                    onChange={(e) => setEmailInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && submitEmail()}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                {/* Validation Error Message */}
                {validationError && (
                  <div className="p-3 bg-red-900/30 border border-red-500/50 rounded-lg flex items-start gap-2">
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-red-300 text-sm">{validationError}</p>
                  </div>
                )}
                
                {/* Submit Button */}
                <button
                  onClick={submitEmail}
                  disabled={emailSubmitting || !firstName.trim() || !emailInput.trim()}
                  className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg font-semibold text-white hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
                >
                  {emailSubmitting ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Submitting...
                    </>
                  ) : (
                    <>
                      Get Free Access
                    </>
                  )}
                </button>
                
                {/* Required Fields Notice */}
                <p className="text-xs text-gray-500 text-center">
                  <span className="text-red-400">*</span> First Name and Email are required
                </p>
                
                {/* Privacy Notice */}
                <p className="text-xs text-gray-400 text-center">
                  We respect your privacy. Unsubscribe anytime. No spam, ever.
                </p>
              </div>
              
              {/* Maybe Later Button */}
              <button
                onClick={() => {
                  setShowEmailModal(false);
                  setValidationError('');
                  setFirstName('');
                  setLastName('');
                  setEmailInput('');
                }}
                className="mt-4 text-gray-400 hover:text-white text-sm transition-colors w-full text-center"
              >
                Maybe later
              </button>
            </div>
          </div>
        )}

        {/* Success Message */}
        {showSuccessMessage && (
          <div className="fixed top-4 right-4 z-50 animate-fadeIn">
            <div className="bg-green-500/90 backdrop-blur-sm text-white px-6 py-3 rounded-lg shadow-lg flex items-center gap-3">
              <div className="w-6 h-6 bg-white rounded-full flex items-center justify-center">
                <span className="text-green-500 font-bold">✓</span>
              </div>
              <div>
                <div className="font-semibold">Access Granted!</div>
                <div className="text-sm text-green-100">You can now analyze unlimited stocks!</div>
                <div className="text-xs text-green-200 mt-1">📧 Welcome email sent - check inbox & spam</div>
              </div>
            </div>
          </div>
        )}

        {/* Welcome Message - Shows before any analysis */}
        {!analysis && !loading && (
          <div className="text-center py-16 px-4">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-2xl font-semibold text-gray-300 mb-4">
                Enter a UK or US stock name/ticker or simply use the dropdown to begin analysis
              </h2>
              <p className="text-gray-400 text-lg">
                Examples: BARC.L (Barclays), BP.L (BP), VOD.L (Vodafone), AAPL (Apple Inc.)
              </p>
            </div>
          </div>
        )}

        <style>{`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
          }
          .animate-fadeIn {
            animation: fadeIn 0.5s ease-out;
          }
        `}</style>

      </div>
    </div>
  );
};

export default ShareAnalyzer;
