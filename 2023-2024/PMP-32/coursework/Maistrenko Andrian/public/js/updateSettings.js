import { showAlert } from "./alerts.js";

// type is password or data
const updateSettings = async (data, type) => {
  try {
    const url =
      type === "password"
        ? "/api/v1/users/updateMyPassword"
        : "/api/v1/users/updateMe";

    let api = new Frisbee({
      baseURI: "", // optional
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, PUT, PATCH, POST, DELETE",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });

    const formData = new FormData();
    if (type === "data") {
      const { name, email, photo } = data;
      const res = await api.patch(url, {
        body: data,
      });

      // const { name, email, photo } = data;
      // formData.append('name', name);
      // formData.append('email', email);
      // formData.append('photo', photo);

      // const res = await api.patch(url, {
      //     body: data
      // });

      if (res.ok) {
        showAlert("success", `${type.toUpperCase()} updated successfully`);
      }
    } else if (type === "password") {
      const { passwordCurrent, password, passwordConfirm } = data;
      const res = await api.patch(url, {
        body: JSON.stringify({
          passwordCurrent: passwordCurrent,
          password: password,
          passwordConfirm: passwordConfirm,
        }),
      });
      // console.log(res.body);
      if (res.ok) {
        showAlert("success", `${type.toUpperCase()} updated successfully`);
      } else {
        showAlert("error", res.err);
      }
    }
  } catch (err) {
    showAlert("Error", err);
  }
};

document
  .querySelector(".form-user-data")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    // console.log("NAME:", document.getElementById("name").value);
    const form = new FormData();
    form.append("name", document.getElementById("name").value);
    form.append("email", document.getElementById("email").value);
    form.append("photo", document.getElementById("photo").files[0]);

    try {
      const response = await fetch("/api/v1/users/updateMe", {
        method: "PATCH",
        body: form,
      });
      const data = await response.json();
      // console.log(data);
    } catch (error) {
      console.error("Error:", error);
    }
    // const name = document.getElementById('name').value;
    // const email = document.getElementById('email').value;
    updateSettings(form, "data");
  });

document
  .querySelector(".form-user-password")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    document.querySelector(".btn--save-password").textContent = "Updating.....";

    const passwordCurrent = document.getElementById("password-current").value;
    const password = document.getElementById("password").value;
    const passwordConfirm = document.getElementById("password-confirm").value;
    await updateSettings(
      { passwordCurrent, password, passwordConfirm },
      "password",
    );

    document.querySelector(".btn--save-password").textContent = "Save password";

    document.getElementById("password-current").value = "";
    document.getElementById("password").value = "";
    document.getElementById("password-confirm").value = "";
  });
