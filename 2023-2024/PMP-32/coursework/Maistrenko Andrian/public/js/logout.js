import { showAlert } from "./alerts.js";

const logout = async () => {
  try {
    let api = new Frisbee({
      baseURI: "", // optional
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, PUT, PATCH, POST, DELETE",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });

    const res = await api.get("/api/v1/users/logout");
    if (res.ok) {
      location.reload(true);
    }
  } catch (err) {
    showAlert("error", "Error loging out");
  }
};

document.querySelector(".nav__el--logout").addEventListener("click", logout);
