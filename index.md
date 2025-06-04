# the-ai-system-design
The AI System Design


No extra configuration required – the CSS that ships with every default theme styles it automatically.

---

### 3 — What about LaTeX equations?  

| Scenario | Does it “just work”? | What to add |
|----------|---------------------|-------------|
| **Markdown inside the repo UI** (README, issues, etc.) | **Yes.** Since May 2022 GitHub renders math natively with MathJax when you view Markdown on github.com. :contentReference[oaicite:4]{index=4} |
| **GitHub Pages site built with Jekyll** | **Not by default.** Pages ships the HTML exactly as Jekyll produced it, so the math delimiters remain raw. | Embed MathJax *or* KaTeX once and you’re done:<br><br>`_config.yml`<br>`markdown: kramdown`<br>`kramdown:`<br>&nbsp;&nbsp;`math_engine: mathjax`<br><br>`_includes/head-custom.html` (or your layout’s `<head>`):<br>`<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>` |
| **Theme that already bundles KaTeX/MathJax** (e.g. *just-the-docs*, *minimal-mistakes*) | Works out of the box once you enable the theme. | Activate the theme in `_config.yml`; write math with `$…$` or `$$…$$`. |

Math delimiters that Pages/Jekyll + MathJax / KaTeX understand:

* Inline: `$E=mc^2$`
* Display:  

  ```math
  \int_{-\infty}^{\infty} e^{-x^{2}}\,dx = \sqrt{\pi}
