import { showAlert } from "./alerts.js";

const login = async (email, password) => {
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
    const res = await api.post("/api/v1/users/login", {
      body: {
        email,
        password,
      },
    });

    if (res.ok) {
      showAlert("success", "Logged in successfully");
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

// const logout = async () => {
//   try {
//     const res = await api.get("/users/logout");
//     if (res.ok) {
//       location.reload(true);
//     }
//   } catch (err) {
//     showAlert("error", "Error loging out");
//   }
// };

document.querySelector(".form--login").addEventListener("submit", (e) => {
  e.preventDefault();
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  login(email, password);
});
