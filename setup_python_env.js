module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "python --version",
        path: "app"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "python validate_python.py",
        path: "app"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "python -m venv venv --python=python3.10",
        path: "app"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "venv",
        path: "app",
        message: "python validate_python.py"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "venv",
        path: "app",
        message: "python -m pip install --upgrade pip setuptools wheel"
      }
    }
  ]
}