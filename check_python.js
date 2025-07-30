module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "python --version",
        path: "app",
        venv: "venv"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "python -c \"import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')\"",
        path: "app",
        venv: "venv"
      }
    }
  ]
}