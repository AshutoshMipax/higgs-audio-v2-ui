module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull",
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
    }
  ]
}
