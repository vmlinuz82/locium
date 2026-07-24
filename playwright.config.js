module.exports = {
  testDir: "./tests/e2e",
  timeout: 30000,
  use: { baseURL: "http://127.0.0.1:7799", headless: true },
  webServer: {
    command: ".venv/bin/python tests/e2e/fixture_server.py",
    url: "http://127.0.0.1:7799",
    reuseExistingServer: false,
    timeout: 120000,
  },
};
