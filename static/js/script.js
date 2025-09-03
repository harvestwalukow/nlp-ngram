document.addEventListener("DOMContentLoaded", function () {
  const healthCheckBtn = document.getElementById("healthCheck");
  const apiResult = document.getElementById("apiResult");

  healthCheckBtn.addEventListener("click", async function () {
    try {
      healthCheckBtn.disabled = true;
      healthCheckBtn.textContent = "Testing...";

      const response = await fetch("/api/health");
      const data = await response.json();

      if (response.ok) {
        apiResult.textContent = `✅ ${data.message}`;
        apiResult.className = "success";
      } else {
        throw new Error("API request failed");
      }
    } catch (error) {
      apiResult.textContent = `❌ Error: ${error.message}`;
      apiResult.className = "error";
    } finally {
      healthCheckBtn.disabled = false;
      healthCheckBtn.textContent = "Test API";
    }
  });
});
