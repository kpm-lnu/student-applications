import { showAlert } from "./alerts.js";

const signup = async (name, email, password, passwordConfirm) => {

  let api = new Frisbee({
    baseURI: "", // optional
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, PUT, PATCH, POST, DELETE",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });

  try {
    const res = await api.post("/api/v1/users/signup", {
      body: {
        name,
        email,
        password,
        passwordConfirm,
      },
    });

    if (res.ok) {
      showAlert("success", "Sign up successfully");
      window.setTimeout(() => {
        location.assign("/");
      }, 1500);
    }
    if (res.err) {
      showAlert("error", res.err);
    }
  } catch (err) {
    showAlert("error", err.responseText);
  }
};

document.querySelector(".form--signup").addEventListener("submit", (e) => {
  e.preventDefault();
  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const passwordConfirm = document.getElementById("passwordConfirm").value;
  signup(name, email, password, passwordConfirm);
});
