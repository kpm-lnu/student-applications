import { showAlert } from "./alerts.js";

const managePurchase = async(data,type) => {
    try {
        let url = "";
        if(type === "create")
        {
            url = "/api/v1/purchases/createPurchase";
        }
        else if(type === "update")
        {
            url = "/api/v1/purchases/updatePurchase";
        }
        else if(type === "delete") 
        {
            url = "/api/v1/purchases/deletePurchase";
        }
    
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
        if (type === "create") {
          const { name, email, photo } = data;
          const res = await api.patch(url, {
            body: data,
          });

          if (res.ok) {
            showAlert("success", `${type.toUpperCase()} updated successfully`);
          }
        } else if (type === "update") {
            const { name, email, photo } = data;
            const res = await api.patch(url, {
              body: data,
            });
        }
        else if (type === "delete") {
            const { purchaseId} = data;
            const res = await api.patch(url, {
                body: data,
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

document.querySelector(".form-purchase-create").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("productId",document.getElementById("productId").value);
    form.append("userId",document.getElementById("userId").value);
    form.append("price",document.getElementById("price").value);

    try {
        const response = await fetch("/api/v1/purchase/createPurchase", {
          method: "PATCH",
          body: form,
        });
        const data = await response.json();
        // console.log(data);
      } catch (error) {
        console.error("Error:", error);
      }
      managePurchase(form, "create");
});

document.querySelector(".form-purchase-update").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("purchaseId",document.getElementById("purchaseId").value);
    form.append("productId",document.getElementById("productId").value);
    form.append("userId",document.getElementById("userId").value);
    form.append("price",document.getElementById("price").value);

    try {
        const response = await fetch("/api/v1/purchase/createPurchase", {
          method: "PATCH",
          body: form,
        });
        const data = await response.json();
        // console.log(data);
      } catch (error) {
        console.error("Error:", error);
      }
      managePurchase(form, "update");
});

document.querySelector(".form-purchase-delete").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append("productId",document.getElementById("productId").value);

    try {
        const response = await fetch("/api/v1/purchase/createPurchase", {
          method: "PATCH",
          body: form,
        });
        const data = await response.json();
        // console.log(data);
      } catch (error) {
        console.error("Error:", error);
      }
      managePurchase(form, "delete");
});