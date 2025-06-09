import { showAlert } from "./alerts.js";

const getStats = async (email) => {
    try {
        const url = '/api/v1/users/getStats';
        let api = new Frisbee({
            baseURI: "", // optional
            headers: {
              "Content-Type": "application/json",
              "Access-Control-Allow-Origin": "*",
              "Access-Control-Allow-Methods": "GET, PUT, PATCH, POST, DELETE",
              "Access-Control-Allow-Headers": "Content-Type",
            },
          });
        const res = await api.post(url, { email: email});
        if (res.ok) {
            showAlert("success", `Stats loaded successfully`);
        }
    } catch (err) {
        showAlert("Error", err);
    }
}

document
  .querySelector(".form-user-stats")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("email", document.getElementById("email").value);

    try {
      const response = await post("/api/v1/users/getStats", {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      // console.log(data);
    } catch (error) {
      console.error("Error:", error);
    }
    getStats(form, "data");
  });