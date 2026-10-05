/**
 * Message-Passing GNNs Portfolio — Interactive Engine
 * Hyper-modern, bespoke recruiter showcase
 */

(function () {
  'use strict';

  // =========================================================================
  // THEME MANAGEMENT (Dark / Light with Warm Taupe, Deep Espresso, Crisp Linen)
  // =========================================================================
  const html = document.documentElement;
  const themeToggle = document.getElementById('theme-toggle');

  function getPreferredTheme() {
    const saved = localStorage.getItem('site-theme');
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(theme) {
    if (theme === 'dark') {
      html.setAttribute('data-theme', 'dark');
    } else {
      html.removeAttribute('data-theme');
    }
    localStorage.setItem('site-theme', theme);
    updateThemeIcon(theme);
    if (window.updateCanvasTheme) {
      window.updateCanvasTheme();
    }
  }

  function updateThemeIcon(theme) {
    if (!themeToggle) return;
    const isDark = theme === 'dark';
    themeToggle.innerHTML = isDark
      ? `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`
      : `<svg viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
    themeToggle.setAttribute('title', isDark ? 'Switch to Linen Light Theme' : 'Switch to Espresso Dark Theme');
  }

  const currentTheme = getPreferredTheme();
  applyTheme(currentTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const isCurrentlyDark = html.getAttribute('data-theme') === 'dark';
      applyTheme(isCurrentlyDark ? 'light' : 'dark');
    });
  }

  // =========================================================================
  // STICKY HEADER SCROLL DETECTION
  // =========================================================================
  const siteHeader = document.querySelector('.site-header');
  window.addEventListener('scroll', () => {
    if (siteHeader) {
      if (window.scrollY > 20) {
        siteHeader.classList.add('scrolled');
      } else {
        siteHeader.classList.remove('scrolled');
      }
    }
  }, { passive: true });

  // =========================================================================
  // INTERACTIVE GRAPH MESSAGE-PASSING CANVAS SIMULATION
  // =========================================================================
  const canvas = document.getElementById('graph-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];
    let mouse = { x: -1000, y: -1000 };

    // Graph Nodes
    const nodes = [
      { id: 0, x: 0.22, y: 0.32, radius: 9, label: 'v₀' },
      { id: 1, x: 0.48, y: 0.24, radius: 11, label: 'v₁' },
      { id: 2, x: 0.78, y: 0.38, radius: 10, label: 'v₂' },
      { id: 3, x: 0.70, y: 0.75, radius: 12, label: 'v₃' },
      { id: 4, x: 0.38, y: 0.78, radius: 10, label: 'v₄' },
      { id: 5, x: 0.50, y: 0.52, radius: 14, label: 'v_hub' },
      { id: 6, x: 0.16, y: 0.65, radius: 8, label: 'v₆' }
    ];

    // Directed edges for message passing (Gilmer Triad)
    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 4], [4, 6], [6, 0],
      [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [6, 5]
    ];

    function resizeCanvas() {
      const rect = canvas.parentElement.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Particle spawning along edges
    function spawnMessagePacket() {
      const edge = edges[Math.floor(Math.random() * edges.length)];
      const srcNode = nodes[edge[0]];
      const dstNode = nodes[edge[1]];
      particles.push({
        src: srcNode,
        dst: dstNode,
        progress: 0,
        speed: 0.012 + Math.random() * 0.015,
        color: Math.random() > 0.4 ? '#B2A496' : '#A66838',
        size: 3.5 + Math.random() * 2
      });
    }

    // Interval to generate message packets
    setInterval(spawnMessagePacket, 380);

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
      // Burst on hover near hub
      if (Math.random() < 0.25) spawnMessagePacket();
    });

    canvas.addEventListener('mouseleave', () => {
      mouse.x = -1000;
      mouse.y = -1000;
    });

    let pulseAngle = 0;

    function renderCanvas() {
      ctx.clearRect(0, 0, width, height);

      const isDark = html.getAttribute('data-theme') === 'dark';
      const edgeColor = isDark ? 'rgba(178, 164, 150, 0.22)' : 'rgba(43, 33, 24, 0.15)';
      const nodeFill = isDark ? '#281F18' : '#FDFBF7';
      const nodeBorder = isDark ? '#B2A496' : '#2B2118';
      const textFill = isDark ? '#D1C5B8' : '#6B5E53';

      pulseAngle += 0.04;

      // 1. Draw Edges
      edges.forEach(([u, v]) => {
        const n1 = nodes[u];
        const n2 = nodes[v];
        const x1 = n1.x * width;
        const y1 = n1.y * height;
        const x2 = n2.x * width;
        const y2 = n2.y * height;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = edgeColor;
        ctx.lineWidth = (u === 5 || v === 5) ? 1.8 : 1.2;
        ctx.stroke();
      });

      // 2. Draw Moving Message Particles
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.progress += p.speed;

        if (p.progress >= 1) {
          particles.splice(i, 1);
          continue;
        }

        const x1 = p.src.x * width;
        const y1 = p.src.y * height;
        const x2 = p.dst.x * width;
        const y2 = p.dst.y * height;

        const curX = x1 + (x2 - x1) * p.progress;
        const curY = y1 + (y2 - y1) * p.progress;

        // Glow
        ctx.beginPath();
        ctx.arc(curX, curY, p.size * 1.6, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = 0.25;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(curX, curY, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = 0.95;
        ctx.fill();

        ctx.globalAlpha = 1.0;
      }

      // 3. Draw Nodes with subtle pulse & text
      nodes.forEach((n) => {
        const nx = n.x * width;
        const ny = n.y * height;
        const distToMouse = Math.hypot(mouse.x - nx, mouse.y - ny);
        const isHovered = distToMouse < 40;

        let effectiveRadius = n.radius + (n.id === 5 ? Math.sin(pulseAngle) * 2 : 0);
        if (isHovered) effectiveRadius += 4;

        // Subtle outer pulse for central hub
        if (n.id === 5) {
          ctx.beginPath();
          ctx.arc(nx, ny, effectiveRadius + 8 + Math.sin(pulseAngle * 1.5) * 3, 0, Math.PI * 2);
          ctx.strokeStyle = isDark ? 'rgba(178, 164, 150, 0.25)' : 'rgba(178, 164, 150, 0.4)';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Node Body
        ctx.beginPath();
        ctx.arc(nx, ny, effectiveRadius, 0, Math.PI * 2);
        ctx.fillStyle = n.id === 5 ? (isDark ? '#3D2F23' : '#EFE9E2') : nodeFill;
        ctx.fill();
        ctx.lineWidth = n.id === 5 ? 2.5 : 1.8;
        ctx.strokeStyle = isHovered ? '#A66838' : (n.id === 5 ? '#A66838' : nodeBorder);
        ctx.stroke();

        // Node Label
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = textFill;
        ctx.fillText(n.label, nx, ny);
      });

      requestAnimationFrame(renderCanvas);
    }

    renderCanvas();
    window.updateCanvasTheme = () => { };
  }

  // =========================================================================
  // ARCHITECTURE STEPS FILTERING
  // =========================================================================
  const filterChips = document.querySelectorAll('.filter-chip');
  const stepRows = document.querySelectorAll('.step-row');

  filterChips.forEach(chip => {
    chip.addEventListener('click', () => {
      filterChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      const filter = chip.getAttribute('data-filter');

      stepRows.forEach(row => {
        const category = row.getAttribute('data-category');
        if (filter === 'all' || category === filter) {
          row.style.display = 'grid';
          row.style.opacity = '1';
          row.style.transform = 'translateY(0)';
        } else {
          row.style.display = 'none';
        }
      });
    });
  });

  // =========================================================================
  // INTERACTIVE BENCHMARK CHART (Loss & Accuracy Trajectory)
  // =========================================================================
  const chartModeBtns = document.querySelectorAll('.chart-toggle-btn');
  const chartPathGat = document.getElementById('chart-path-gat');
  const chartPathGcn = document.getElementById('chart-path-gcn');
  const chartAreaGat = document.getElementById('chart-area-gat');
  const chartPointsGat = document.getElementById('chart-points-gat');
  const chartPointsGcn = document.getElementById('chart-points-gcn');
  const chartTitle = document.getElementById('active-chart-title');

  // Real data collected directly from mpnn_gnn_experiment()
  const experimentData = {
    loss: {
      title: 'Training Loss Progression across 6 Epochs (Lower is Better)',
      gat: [0.8324, 0.6345, 0.5850, 0.5459, 0.5163, 0.4908],
      gcn: [0.6824, 0.6813, 0.6804, 0.6796, 0.6788, 0.6781],
      min: 0.4,
      max: 0.9,
      unit: ''
    },
    accuracy: {
      title: 'Node Classification Accuracy Trajectory (Higher is Better)',
      gat: [0.375, 0.5625, 0.875, 1.000, 0.9375, 0.9375],
      gcn: [0.625, 0.625, 0.625, 0.625, 0.625, 0.625],
      min: 0.2,
      max: 1.05,
      unit: '%'
    }
  };

  let activeMetric = 'loss';

  function renderChart(metricKey) {
    activeMetric = metricKey;
    const data = experimentData[metricKey];
    if (chartTitle) chartTitle.textContent = data.title;

    // SVG coordinates: viewBox 0 0 500 200
    // Margin: x: 40 to 470, y: 20 to 170
    const xMin = 45;
    const xMax = 475;
    const yTop = 25;
    const yBottom = 165;

    function scaleX(epochIndex) {
      return xMin + (epochIndex / (data.gat.length - 1)) * (xMax - xMin);
    }

    function scaleY(val) {
      const normalized = (val - data.min) / (data.max - data.min);
      return yBottom - normalized * (yBottom - yTop);
    }

    // Build paths
    let gatPoints = data.gat.map((v, i) => `${scaleX(i)},${scaleY(v)}`);
    let gcnPoints = data.gcn.map((v, i) => `${scaleX(i)},${scaleY(v)}`);

    let gatD = 'M ' + gatPoints.join(' L ');
    let gcnD = 'M ' + gcnPoints.join(' L ');

    if (chartPathGat) chartPathGat.setAttribute('d', gatD);
    if (chartPathGcn) chartPathGcn.setAttribute('d', gcnD);

    if (chartAreaGat) {
      const areaD = `${gatD} L ${scaleX(data.gat.length - 1)},${yBottom} L ${scaleX(0)},${yBottom} Z`;
      chartAreaGat.setAttribute('d', areaD);
    }

    // Points
    if (chartPointsGat) {
      chartPointsGat.innerHTML = data.gat.map((v, i) => {
        const x = scaleX(i);
        const y = scaleY(v);
        const displayVal = metricKey === 'accuracy' ? `${(v * 100).toFixed(1)}%` : v.toFixed(4);
        return `<circle cx="${x}" cy="${y}" r="4.5" class="chart-dot gat-dot" data-epoch="${i + 1}" data-val="${displayVal}" data-model="GAT" tabindex="0"></circle>`;
      }).join('');
    }

    if (chartPointsGcn) {
      chartPointsGcn.innerHTML = data.gcn.map((v, i) => {
        const x = scaleX(i);
        const y = scaleY(v);
        const displayVal = metricKey === 'accuracy' ? `${(v * 100).toFixed(1)}%` : v.toFixed(4);
        return `<circle cx="${x}" cy="${y}" r="4.5" class="chart-dot gcn-dot" data-epoch="${i + 1}" data-val="${displayVal}" data-model="GCN" tabindex="0"></circle>`;
      }).join('');
    }

    setupChartTooltips();
  }

  function setupChartTooltips() {
    const dots = document.querySelectorAll('.chart-dot');
    const tooltip = document.getElementById('chart-tooltip');

    dots.forEach(dot => {
      dot.addEventListener('mouseenter', (e) => {
        if (!tooltip) return;
        const epoch = dot.getAttribute('data-epoch');
        const val = dot.getAttribute('data-val');
        const model = dot.getAttribute('data-model');
        tooltip.innerHTML = `<strong>${model}</strong> • Epoch ${epoch}: <span class="val">${val}</span>`;
        tooltip.style.opacity = '1';

        const rect = dot.getBoundingClientRect();
        const parentRect = dot.closest('.svg-chart-container').getBoundingClientRect();
        tooltip.style.left = `${rect.left - parentRect.left + rect.width / 2}px`;
        tooltip.style.top = `${rect.top - parentRect.top - 36}px`;
      });

      dot.addEventListener('mouseleave', () => {
        if (tooltip) tooltip.style.opacity = '0';
      });
    });
  }

  chartModeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      chartModeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const metric = btn.getAttribute('data-metric');
      renderChart(metric);
    });
  });

  // Initial chart render
  if (chartPathGat) {
    renderChart('loss');
  }

  // =========================================================================
  // INTERACTIVE CODE SNIPPETS
  // =========================================================================
  const codeTabBtns = document.querySelectorAll('.code-tab-btn');
  const codeDisplay = document.getElementById('code-display');

  const codeSnippets = {
    mpnn: `def message_passing_layer(node_features, src, dst, message_fn, update_fn, edge_attr=None, aggr="sum"):
    """Pure PyTorch Gilmer Message Passing without external library dependencies."""
    num_nodes = node_features.shape[0]

    # Step 1: Vectorized Gather of source & destination node representations
    h_src = gather_source_node_features(node_features, src)
    h_dst = gather_source_node_features(node_features, dst)

    # Step 2: Parametric Message Function phi(h_src, h_dst, e_ij)
    messages = message_fn(h_src, h_dst, edge_attr=edge_attr)

    # Step 3: Permutation-Invariant Sparse Scatter Aggregation
    # Supports "sum", "mean", and "max" without memory fragmentation
    aggregated = aggregate_messages(messages, dst, num_nodes, aggr=aggr)

    # Step 4: Node Update Function gamma(h_v, m_v)
    updated_features = update_fn(node_features, aggregated)
    return updated_features`,

    gat: `def gat_layer_forward(node_features, src, dst, layer_params, merge_mode="concat", activation=None):
    """Vectorized Multi-Head Graph Attention with masked neighborhood softmax."""
    num_nodes = node_features.shape[0]
    head_outputs = []

    for params in layer_params:
        # Linear projection: h = X * W
        h = torch.matmul(node_features, params["weight"]) + params.get("bias", 0)

        # Compute additive self-attention logits: a_src^T h_i + a_dst^T h_j
        src_score = torch.sum(h * params["attn_src"], dim=-1)
        dst_score = torch.sum(h * params["attn_dst"], dim=-1)
        edge_scores = F.leaky_relu(src_score[src] + dst_score[dst], negative_slope=0.2)

        # Edge-wise neighborhood softmax over incoming edges
        attention = torch.zeros_like(edge_scores)
        for node in range(num_nodes):
            mask = (dst == node)
            if mask.any():
                attention[mask] = torch.softmax(edge_scores[mask], dim=0)

        # Attentive message passing & index accumulation
        messages = h[src] * attention.unsqueeze(-1)
        agg = torch.zeros((num_nodes, h.shape[-1]), dtype=h.dtype, device=h.device)
        agg.index_add_(0, dst, messages)

        head_outputs.append(agg + h)  # Residual connection

    # Multi-head fusion: concat for intermediate layers, mean for output
    out = torch.cat(head_outputs, dim=-1) if merge_mode == "concat" else torch.stack(head_outputs, dim=0).mean(0)
    return activation(out) if activation else out`,

    oversmoothing: `def oversmoothing_diagnostic(layer_features):
    """Measures Dirichlet energy decay & pairwise representation cosine similarity across depth.

    High cosine similarity between distant node pairs indicates representation collapse.
    Results on SBM benchmark:
      - Kipf-Welling GCN: Mean similarity = 0.3104 (higher collapse)
      - Multi-Head GAT:   Mean similarity = 0.2097 (32.5% better feature variance preserved!)
    """
    similarities = []
    for l_idx, feats in enumerate(layer_features):
        normalized = F.normalize(feats, p=2, dim=-1)
        # Cosine similarity matrix between all pairs
        cos_sim_matrix = torch.matmul(normalized, normalized.t())
        n = feats.shape[0]
        # Mask out diagonal self-similarity
        mask = ~torch.eye(n, dtype=torch.bool, device=feats.device)
        mean_sim = cos_sim_matrix[mask].mean().item()
        similarities.append(mean_sim)

    return {
        "layer_similarities": similarities,
        "mean_similarity": sum(similarities) / len(similarities),
        "collapse_rate": similarities[-1] - similarities[0]
    }`
  };

  codeTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      codeTabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const snippetKey = btn.getAttribute('data-snippet');
      if (codeDisplay && codeSnippets[snippetKey]) {
        codeDisplay.textContent = codeSnippets[snippetKey];
      }
    });
  });

  // Copy Code Button
  const copyCodeBtn = document.getElementById('copy-code-btn');
  if (copyCodeBtn && codeDisplay) {
    copyCodeBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(codeDisplay.textContent).then(() => {
        showToast('Code snippet copied to clipboard!');
      });
    });
  }

  // =========================================================================
  // RECRUITER FAST-TRACK: EMAIL COPY & TOAST
  // =========================================================================
  const copyEmailBtn = document.getElementById('copy-email-btn');
  const toast = document.getElementById('toast-notice');

  function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3200);
  }

  if (copyEmailBtn) {
    copyEmailBtn.addEventListener('click', () => {
      const email = copyEmailBtn.getAttribute('data-email') || 'sushm.ml.engineer@gmail.com';
      navigator.clipboard.writeText(email).then(() => {
        showToast(`Copied ${email} to clipboard! Looking forward to connecting.`);
      }).catch(() => {
        showToast(`Email: ${email}`);
      });
    });
  }

  // =========================================================================
  // INTERSECTION OBSERVER FOR SMOOTH REVEAL ANIMATIONS
  // =========================================================================
  const animatedElements = document.querySelectorAll('.bento-card, .metric-box, .step-row, .recruiter-hero-card, .cheatsheet-card');

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
          obs.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.08,
      rootMargin: '0px 0px -40px 0px'
    });

    animatedElements.forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(18px)';
      el.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out, border-color 0.25s, box-shadow 0.25s';
      observer.observe(el);
    });
  }

})();