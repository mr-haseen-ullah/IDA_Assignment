import { useState, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from '@/components/ui/button';
import { Download, TrendingUp, Users, ShoppingCart, MapPin } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

/**
 * Design Philosophy: Modern Data-Driven Minimalism
 * - Clean, uncluttered layouts emphasizing data clarity
 * - Deep slate blue (#1e293b) primary with vibrant teal (#0891b2) accents
 * - Generous whitespace and subtle animations
 * - Poppins Bold for headings, Inter for body text
 */

// Sample data for visualizations
const monthlyRevenueData = [
  { month: 'Jan', revenue: 45000, profit: 12000 },
  { month: 'Feb', revenue: 52000, profit: 15000 },
  { month: 'Mar', revenue: 48000, profit: 13000 },
  { month: 'Apr', revenue: 61000, profit: 18000 },
  { month: 'May', revenue: 55000, profit: 16000 },
  { month: 'Jun', revenue: 67000, profit: 20000 },
];

const categoryPerformance = [
  { category: 'Electronics', revenue: 125000, profit: 35000 },
  { category: 'Furniture', revenue: 98000, profit: 25000 },
  { category: 'Clothing', revenue: 87000, profit: 18000 },
  { category: 'Home & Garden', revenue: 76000, profit: 15000 },
];

const paymentModes = [
  { mode: 'Credit Card', value: 45, revenue: 185000 },
  { mode: 'Debit Card', value: 30, revenue: 123000 },
  { mode: 'UPI', value: 15, revenue: 61000 },
  { mode: 'Cash', value: 10, revenue: 41000 },
];

const statePerformance = [
  { state: 'California', revenue: 156000, orders: 1240 },
  { state: 'Texas', revenue: 142000, orders: 1120 },
  { state: 'Florida', revenue: 128000, orders: 980 },
  { state: 'New York', revenue: 115000, orders: 890 },
  { state: 'Illinois', revenue: 98000, orders: 760 },
];

const COLORS = ['#0891b2', '#06b6d4', '#22d3ee', '#67e8f9'];

export default function Home() {
  const [activeTab, setActiveTab] = useState('overview');

  // Key metrics
  const metrics = useMemo(() => ({
    totalRevenue: 410000,
    totalProfit: 97000,
    totalCustomers: 2840,
    avgOrderValue: 144,
  }), []);

  const handleExport = () => {
    const report = `
IDA Retail Analysis Dashboard Report
Generated: ${new Date().toLocaleDateString()}

KEY METRICS:
- Total Revenue: $${metrics.totalRevenue.toLocaleString()}
- Total Profit: $${metrics.totalProfit.toLocaleString()}
- Total Customers: ${metrics.totalCustomers.toLocaleString()}
- Average Order Value: $${metrics.avgOrderValue}

INSIGHTS:
1. Monthly Revenue Trend: Shows consistent growth with peak in June
2. Category Performance: Electronics leads with $125K revenue
3. Payment Mode Distribution: Credit cards dominate at 45% of transactions
4. Geographic Distribution: California and Texas are top performers
    `;

    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(report));
    element.setAttribute('download', 'retail-analysis-report.txt');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="hero-section py-16 md:py-24">
        <div className="hero-bg" />
        <div className="container relative z-10">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
              Retail Data Analysis Dashboard
            </h1>
            <div className="flex items-center gap-4 mb-8 p-4 bg-secondary/50 rounded-lg backdrop-blur-sm border border-border w-fit">
              <Avatar className="h-16 w-16 border-2 border-accent">
                <AvatarImage src="https://github.com/mr-haseen-ullah.png" alt="Haseen ullah" />
                <AvatarFallback>HU</AvatarFallback>
              </Avatar>
              <div>
                <h2 className="text-xl font-bold text-foreground">Haseen ullah</h2>
                <p className="text-sm text-muted-foreground">Reg. No: 22MDSWE238</p>
              </div>
            </div>

            <p className="text-lg text-muted-foreground mb-8">
              Comprehensive insights into sales performance, customer behavior, and market trends. Explore data intuitively, understand trends better, and easily save or share your findings.
            </p>
            <div className="flex gap-4">
              <Button
                onClick={handleExport}
                className="bg-accent hover:bg-accent/90 text-accent-foreground gap-2"
              >
                <Download className="w-4 h-4" />
                Export Report
              </Button>
              <Button
                variant="outline"
                className="border-border text-foreground hover:bg-secondary"
              >
                View Documentation
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="py-12">
        <div className="container">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Total Revenue</p>
                  <p className="metric-value">${(metrics.totalRevenue / 1000).toFixed(0)}K</p>
                </div>
                <ShoppingCart className="w-8 h-8 text-accent opacity-20" />
              </div>
              <p className="text-xs text-muted-foreground mt-2">↑ 12% from last period</p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Total Profit</p>
                  <p className="metric-value">${(metrics.totalProfit / 1000).toFixed(0)}K</p>
                </div>
                <TrendingUp className="w-8 h-8 text-accent opacity-20" />
              </div>
              <p className="text-xs text-muted-foreground mt-2">↑ 8% from last period</p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Total Customers</p>
                  <p className="metric-value">{(metrics.totalCustomers / 1000).toFixed(1)}K</p>
                </div>
                <Users className="w-8 h-8 text-accent opacity-20" />
              </div>
              <p className="text-xs text-muted-foreground mt-2">↑ 5% from last period</p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="metric-label">Avg Order Value</p>
                  <p className="metric-value">${metrics.avgOrderValue}</p>
                </div>
                <MapPin className="w-8 h-8 text-accent opacity-20" />
              </div>
              <p className="text-xs text-muted-foreground mt-2">↓ 2% from last period</p>
            </div>
          </div>

          {/* Charts Section */}
          <div className="space-y-8">
            {/* Monthly Revenue Trend */}
            <div className="chart-container">
              <h2 className="text-2xl font-bold text-foreground mb-4">Monthly Revenue & Profit Trend</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={monthlyRevenueData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="month" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0' }}
                    formatter={(value) => `$${value.toLocaleString()}`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="revenue"
                    stroke="#0891b2"
                    strokeWidth={2}
                    dot={{ fill: '#0891b2', r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="profit"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    dot={{ fill: '#06b6d4', r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Category Performance & Payment Modes */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="chart-container">
                <h2 className="text-2xl font-bold text-foreground mb-4">Revenue by Category</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={categoryPerformance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="category" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0' }}
                      formatter={(value) => `$${value.toLocaleString()}`}
                    />
                    <Bar dataKey="revenue" fill="#0891b2" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-container">
                <h2 className="text-2xl font-bold text-foreground mb-4">Payment Mode Distribution</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={paymentModes}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ mode, value }) => `${mode}: ${value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {paymentModes.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => `${value}%`} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Geographic Performance */}
            <div className="chart-container">
              <h2 className="text-2xl font-bold text-foreground mb-4">Top States by Revenue</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={statePerformance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" stroke="#64748b" />
                  <YAxis dataKey="state" type="category" stroke="#64748b" width={100} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0' }}
                    formatter={(value) => `$${value.toLocaleString()}`}
                  />
                  <Bar dataKey="revenue" fill="#06b6d4" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Insights Section */}
          <div className="mt-12 p-8 bg-gradient-to-r from-secondary/10 to-accent/10 rounded-lg border border-border accent-border">
            <h2 className="text-2xl font-bold text-foreground mb-4">Key Insights & Recommendations</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-bold text-primary mb-2">Sales Performance</h3>
                <p className="text-sm text-muted-foreground">
                  Revenue shows consistent growth with a peak in June. Electronics category leads with $125K in revenue, suggesting strong market demand for tech products.
                </p>
              </div>
              <div>
                <h3 className="font-bold text-primary mb-2">Customer Insights</h3>
                <p className="text-sm text-muted-foreground">
                  With 2,840 total customers and an average order value of $144, focus on customer retention and loyalty programs to increase repeat purchases.
                </p>
              </div>
              <div>
                <h3 className="font-bold text-primary mb-2">Payment Optimization</h3>
                <p className="text-sm text-muted-foreground">
                  Credit cards dominate at 45% of transactions. Consider promoting alternative payment methods to diversify revenue streams and reduce transaction fees.
                </p>
              </div>
              <div>
                <h3 className="font-bold text-primary mb-2">Geographic Expansion</h3>
                <p className="text-sm text-muted-foreground">
                  California and Texas are top performers. Replicate successful strategies from these states in underperforming regions to drive growth.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 mt-16">
        <div className="container">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground">
              © 2025 IDA Retail Analysis Dashboard. All rights reserved.
            </p>
            <div className="flex gap-6">
              <a href="#" className="text-sm text-muted-foreground hover:text-accent transition-colors">
                Documentation
              </a>
              <a href="#" className="text-sm text-muted-foreground hover:text-accent transition-colors">
                GitHub
              </a>
              <a href="#" className="text-sm text-muted-foreground hover:text-accent transition-colors">
                Contact
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
