# HiBayES Documentation

This directory contains the Quarto documentation for HiBayES.

## Local Development

### Prerequisites

Install HiBayES with documentation dependencies:
   ```bash
   # From source (development)
   pip install -e ".[docs]"
   ```

### Building Locally

1. **Preview with live reload:**
   ```bash
   cd docs/
   quarto preview
   ```
   This will open the documentation at `http://localhost:4040` with live reload.

2. **Build static site:**
   ```bash
   cd docs/
   quarto render
   ```
   The built site will be in `docs/_site/`.

3. **Serve built site:**
   ```bash
   cd docs/_site/
   python -m http.server 8000
   ```
   Access at `http://localhost:8000`.

### Quick Commands

```bash
# Preview docs with live reload
make docs-dev

# Build docs
make docs-build

# Clean built files
make docs-clean
```

### File Structure

```
docs/
├── _quarto.yml          # Quarto configuration
├── styles.css           # Custom styling
├── index.qmd            # Homepage
├── getting-started.qmd  # Getting started guide
├── workflow.qmd         # Workflow documentation
├── loaders.qmd          # Data loaders
├── processors.qmd       # Data processors
├── models.qmd           # Statistical models
├── communicators.qmd    # Visualisations
├── examples/            # Example notebooks
├── api/                 # API reference
└── _site/               # Built site (generated)
```

### Adding Content

1. **New page:** Create a new `.qmd` file and add to `_quarto.yml` navigation
2. **New example:** Add to `examples/` directory
3. **API docs:** Add to `api/` directory

### Styling

- Main styles in `styles.css`
- Uses Cosmo theme with custom enhancements
- Responsive design with mobile support

### Tips

- Use `quarto preview` for development - it has live reload
- Check `_quarto.yml` for navigation structure
- Test on mobile devices using browser dev tools
