"""
Portfolio Optimization - Complete Working Backend
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import traceback
from numpy.linalg import det

# Try to import your modules
try:
    import act
    import func
    try:
        import indicators
    except:
        indicators = None
    print("✓ Successfully imported your modules (act, func)")
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Could not import your modules: {e}")
    print("⚠ Using mock functions")
    MODULES_AVAILABLE = False
    
    # Mock implementations
    class MockAct:
        @staticmethod
        def initialize_evolution_matrices(n, rho):
            return np.eye(2*n+rho), np.eye(2*n+rho), np.eye(rho)
        
        @staticmethod
        def get_loss_matrix(n, rho, D, sigma, omega):
            return np.eye(2*n+rho) * 0.01
        
        @staticmethod
        def update_evolution_matrix(n, rho, P, X_mat, A):
            return A
        
        @staticmethod
        def action_generation(n, rho, s, B, q, A_update, horizon, H_, P, gxx, sigma_inv, sampling=False):
            a = np.ones(n) / n + np.random.randn(n) * 0.02
            a = np.abs(a)
            a = a / a.sum()
            return a, H_
    
    class MockFunc:
        @staticmethod
        def get_omega(D, sigma, a_):
            return sigma
        
        @staticmethod
        def opt_forget_factors(args):
            return 0.09, 0.05, 0.86
        
        @staticmethod
        def update_G(g, d, g_0, alpha, beta):
            gamma = 1 - alpha - beta
            return np.sqrt(alpha+beta)*g + np.sqrt(beta)*np.outer(d,d) + np.sqrt(gamma)*g_0
        
        @staticmethod
        def optimize_H(phi_0, h_, h):
            return (1-phi_0)*h + phi_0*h_
        
        @staticmethod
        def sample_matrix_normal(M, U, V, size=1):
            p, q = M.shape
            samples = []
            for _ in range(size):
                X = M + np.random.randn(p, q) * 0.01
                samples.append(X)
            return samples if size > 1 else samples[0]
    
    act = MockAct()
    func = MockFunc()

app = Flask(__name__)
CORS(app)

STATS_DIR = 'saved_statistics'
os.makedirs(STATS_DIR, exist_ok=True)

MODEL_PARAMETERS = {
    'mu': 0.9,
    'alpha0': 0.09,
    'beta0': 0.05,
    'phi0': 0.1,
    'D': (2e-4 * np.eye(2)).tolist(),
    'G_0': (1e-3 * np.eye(4)).tolist(),
    'delta0': 14,
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return super().default(obj)

@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modules_available': MODULES_AVAILABLE
    })

@app.route('/api/statistics/list', methods=['GET'])
def list_statistics():
    try:
        files = [f for f in os.listdir(STATS_DIR) if f.endswith('.json')]
        stats_list = []
        for filename in files:
            try:
                with open(os.path.join(STATS_DIR, filename), 'r') as f:
                    data = json.load(f)
                    stats_list.append({
                        'filename': filename,
                        'name': data.get('name', filename),
                        'timestamp': data.get('timestamp', ''),
                        'n_assets': data.get('n_assets', 0),
                        'n_regressors': data.get('n_regressors', 0)
                    })
            except:
                continue
        return jsonify({'success': True, 'statistics': stats_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics/save', methods=['POST'])
def save_statistics():
    try:
        data = request.json
        name = data.get('name', f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        filename = f"{name.replace(' ', '_')}.json"
        
        with open(os.path.join(STATS_DIR, filename), 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        
        return jsonify({'success': True, 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics/load', methods=['POST'])
def load_statistics():
    try:
        filename = request.json.get('filename')
        filepath = os.path.join(STATS_DIR, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return jsonify({'success': True, 'statistics': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics/initialize', methods=['POST'])
def initialize_statistics():
    try:
        n = request.json.get('n_assets', 2)
        rho = request.json.get('n_regressors', 2)
        tickers = request.json.get('tickers', [])
        
        stats = {
            'n_assets': n,
            'n_regressors': rho,
            'G': (1e-2 * np.eye(n + rho)).tolist(),
            'G0': (1e-3 * np.eye(n + rho)).tolist(),
            'Delta0': 14.0,
            'H': (1e-1 * np.eye(3*n + rho)).tolist(),
            'Delta': 5.0,
            'a_prev': (np.ones(n) / n).tolist(),
            'tickers': tickers
        }
        
        return jsonify({'success': True, 'statistics': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/run', methods=['POST'])
def run_model():
    try:
        data = request.json
        g = np.array(data['G'])
        h = np.array(data['H'])
        delta = float(data['Delta'])
        a_prev = np.array(data['a_prev'])
        x = np.array(data['regressor'])
        n = int(data['n_assets'])

        # Run model
        params = {
            'mu': 0.9,
            'alpha0': 0.09,
            'beta0': 0.05,
            'phi0': 0.1,
            'D': 2e-4 * np.eye(n),
            'G_0': np.array(data['G0']),
            'delta0': float(data['Delta0']),
        }
        
        opt_a, H_new = model_run(params, g, x, delta, a_prev, h, n)
        
        # Calculate expected return
        gyy = g[:n, :n]
        gyx = g[:n, n:]
        gxx = g[n:, n:]
        P = gyx @ np.linalg.inv(gxx)
        expected_returns = P @ x
        portfolio_return = np.dot(opt_a, expected_returns)
        
        return jsonify({
            'success': True,
            'optimal_allocation': opt_a.tolist(),
            'H_new': H_new.tolist(),
            'expected_returns': expected_returns.tolist(),
            'portfolio_return': float(portfolio_return)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/model/simulate', methods=['POST'])
def simulate_model():
    try:
        data = request.json
        g_init = np.array(data['G'])
        x_init = np.array(data['regressor'])
        regressor_meta = data.get('regressor_meta', [])
        delta_init = float(data['Delta'])
        n = int(data['n_assets'])
        n_steps = int(data.get('n_steps', 10))
        n_paths = int(data.get('n_paths', 5))

        # Calculate static matrices ONCE outside the simulation loops
        gyy = g_init[:n, :n]
        gyx = g_init[:n, n:]
        gxx = g_init[n:, n:]
        P = gyx @ np.linalg.inv(gxx)

        sigma = 1 / delta_init * (gyy @ gyy.T)
        sigma_inv = np.linalg.inv(sigma)  # Correct inverse covariance

        paths = []
        for _ in range(n_paths):
            path_returns = []
            x_current = x_init.copy()

            for step in range(n_steps):
                # 1. Sample P
                P_sample = func.sample_matrix_normal(P, sigma_inv, gxx)

                # 2. Generate observation
                y_sample = P_sample @ x_current
                path_returns.append(y_sample.tolist())

                # 3. Dynamically update the Regressor (x_next)
                x_next = x_current.copy()
                for i, meta in enumerate(regressor_meta):
                    if meta['type'] == 'return':
                        if meta['lag'] == 0:
                            x_next[i] = y_sample[meta['asset']]
                        else:
                            # Shift previous lags down for this specific asset
                            prev_idx = next(j for j, m in enumerate(regressor_meta)
                                            if m['type'] == 'return' and
                                            m['lag'] == meta['lag'] - 1 and
                                            m['asset'] == meta['asset'])
                            x_next[i] = x_current[prev_idx]

                    elif meta['type'] == 'sine':
                        t_new = meta['t'] + step + 1
                        x_next[i] = np.sin(2 * np.pi * t_new / meta['period'])

                    elif meta['type'] == 'cosine':
                        t_new = meta['t'] + step + 1
                        x_next[i] = np.cos(2 * np.pi * t_new / meta['period'])

                x_current = x_next

            paths.append(path_returns)

        return jsonify({
            'success': True,
            'paths': paths,
            'n_steps': n_steps,
            'n_paths': n_paths
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/update', methods=['POST'])
def update_model():
    try:
        data = request.json
        g = np.array(data['G'])
        h = np.array(data['H'])
        h_new = np.array(data['H_new'])
        delta = float(data['Delta'])
        y = np.array(data['observed_returns'])
        x = np.array(data['regressor'])

        params = {
            'mu': 0.9,
            'alpha0': 0.09,
            'beta0': 0.05,
            'phi0': 0.1,
            'D': 2e-4 * np.eye(len(y)),
            'G_0': np.array(data['G0']),
            'delta0': float(data['Delta0']),
        }
        
        g_update, delta_update, h_update = update(params, g, y, x, delta, h, h_new)
        
        return jsonify({
            'success': True,
            'G_update': g_update.tolist(),
            'Delta_update': float(delta_update),
            'H_update': h_update.tolist()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/indicators/macd', methods=['POST'])
def calculate_macd():
    try:
        prices = np.array(request.json['prices'])
        
        if len(prices) < 10:
            return jsonify({'success': False, 'error': 'Need at least 10 prices'}), 400
        
        # Simple MACD
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = [data[0]]
            for price in data[1:]:
                result.append(alpha * price + (1 - alpha) * result[-1])
            return result[-1]
        
        ema12 = ema(prices, 5)
        ema26 = ema(prices, 10)
        macd = ema12 - ema26
        
        return jsonify({'success': True, 'macd': float(macd)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/correlation', methods=['POST'])
def get_correlation():
    try:
        data = request.json
        g = np.array(data['G'])
        delta = float(data['Delta'])
        n = int(data['n_assets'])

        # Calculate covariance matrix Sigma
        # Formula: Sigma = (1/delta) * (G_yy * G_yy^T)
        gyy = g[:n, :n]
        sigma = (1 / delta) * (gyy @ gyy.T)

        # Derive correlation matrix: corr = sigma / (std_i * std_j)
        d = np.sqrt(np.diag(sigma))
        # Add small epsilon to prevent division by zero if an asset has 0 variance
        correlation = sigma / np.outer(d, d)

        # Clip to ensure floating point math stays within [-1, 1]
        correlation = np.clip(correlation, -1.0, 1.0)

        return jsonify({
            'success': True,
            'correlation': correlation.tolist(),
            'sigma': sigma.tolist()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# API calls function
@app.route('/api/data/fetch', methods=['POST'])
def fetch_market_data():
    try:
        tickers = request.json.get('tickers', [])
        if not tickers:
            return jsonify({'success': False, 'error': 'No tickers provided'}), 400

        df = yf.download(tickers, period="15d", progress=False)['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(name=tickers[0])

        valid_tickers = [t for t in tickers if t in df.columns]
        if valid_tickers:
            df = df[valid_tickers]

        mean_prices = df.mean(axis=1).dropna().tail(10).tolist()
        returns_df = df.pct_change().dropna()
        recent_returns = returns_df.iloc[::-1].head(10).fillna(0).values.tolist()

        return jsonify({
            'success': True,
            'returns': recent_returns,
            'mean_prices': mean_prices
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Model functions
def model_run(params, g, x, delta, a_, H_, n):
    rho = len(x)
    A, B, X_mat = act.initialize_evolution_matrices(n, rho)
    
    gyy = g[:n, :n]
    gyx = g[:n, n:]
    gxx = g[n:, n:]
    
    P = gyx @ np.linalg.inv(gxx)
    y_hat = P @ x
    sigma = 1/delta * (gyy @ gyy.T)
    omega = func.get_omega(params['D'], sigma, a_)
    q = act.get_loss_matrix(n, rho, params['D'], sigma, omega)
    s = np.hstack((a_, np.zeros(n), y_hat, x))
    A_update = act.update_evolution_matrix(n, rho, P, X_mat, A)
    a, H = act.action_generation(n, rho, s, B, q, A_update, 1, H_, P, gxx, np.linalg.inv(sigma), sampling=False)
    
    return a, H

def update(params, g, y, x, delta, h_, h):
    gamma0 = 1 - params['alpha0'] - params['beta0']
    n = len(y)
    
    gyy_0 = params['G_0'][:n, :n]
    gxx_0 = params['G_0'][n:, n:]
    gyy = g[:n, :n]
    gyx = g[:n, n:]
    gxx = g[n:, n:]
    
    P = gyx @ np.linalg.inv(gxx)
    y_hat = P @ x
    e_hat = y - y_hat
    d = np.concatenate((y, x))
    
    args = (params['alpha0'], params['beta0'], gamma0, params['delta0']+1, params['delta0'],
            gxx, det(gxx), gxx_0, det(gxx_0), gyy, det(gyy), gyy_0, det(gyy_0),
            g@g.T, g@g.T+np.outer(d,d), params['G_0']@params['G_0'].T,
            gyy@gyy.T, gyy@gyy.T+np.outer(e_hat,e_hat)/(1+x.T@np.linalg.inv(gxx@gxx.T)@x),
            gyy_0@gyy_0.T, x.T@np.linalg.inv(gxx@gxx.T)@x, e_hat.T@np.linalg.inv(gyy@gyy.T)@e_hat)
    
    alpha, beta, gamma = func.opt_forget_factors(args)
    g_update = func.update_G(g, d, params['G_0'], alpha, beta)
    delta_update = (alpha+beta)*delta + beta + gamma*params['delta0']
    h_update = func.optimize_H(params['phi0'], h_, h)
    
    return g_update, delta_update, h_update

if __name__ == '__main__':
    print("="*60)
    print("Portfolio Optimization Server")
    print("="*60)
    print(f"Modules available: {MODULES_AVAILABLE}")
    print("Open: http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
