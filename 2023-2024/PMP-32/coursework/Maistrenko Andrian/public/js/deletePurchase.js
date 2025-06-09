// const deletePurchase = async(data) => {
//     try {
//         let url = "";
//         url = `/api/v1/purchases/${data.body.purchaseId}`;    
//         let api = new Frisbee({
//           baseURI: "", // optional
//           headers: {
//             "Content-Type": "application/json",
//             "Access-Control-Allow-Origin": "*",
//             "Access-Control-Allow-Methods": "GET, PUT, PATCH, POST, DELETE",
//             "Access-Control-Allow-Headers": "Content-Type",
//           },
//         });
    
//         const res = await api.patch(url, {
//             body: data,
//         });
//         if (res.ok) {
//             alert("success", `${type.toUpperCase()} updated successfully`);
//         } else {
//             alert("error", res.err);
//         }
//     }catch (error) {
//         console.error("Error:", error);
//     }
// };

// document.querySelector(".form-purchase-delete").addEventListener("submit", async (e) => {
//     e.preventDefault();
//     const form = new FormData();
//     form.append("purchaseId",document.getElementById("purchaseId").value);

//     try {
//         const response = await fetch("/api/v1/purchase/createPurchase", {
//           method: "PATCH",
//           body: form,
//         });
//         // const data = await response.json();
//         // console.log(data);
//       } catch (error) {
//         console.error("Error:", error);
//       }
//       deletePurchase(form);
// });