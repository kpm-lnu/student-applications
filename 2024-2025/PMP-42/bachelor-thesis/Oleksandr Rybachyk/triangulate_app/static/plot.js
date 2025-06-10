document.getElementById("upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("file-input");
    const subsetSizeInput = document.getElementById("subset-size");
    const subsetSize = parseInt(subsetSizeInput.value);

    if (isNaN(subsetSize)) {
        alert("Будь ласка, введіть коректне число для розміру підмножини");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("subset_size", subsetSize);

    const loader = document.getElementById("loader");
    const statusText = document.getElementById("status");
    loader.style.display = "block";
    statusText.innerText = "⏳ Обробка... Зачекайте, будь ласка.";

    console.log("Файл завантажено, відправляємо запит...");

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        console.log("Запит виконано, отримано відповідь.");

        if (!response.ok) {
            throw new Error("Помилка при обробці файлу");
        }

        const data = await response.json();
        const points = data.points;
        const triangles = data.triangles;

        const x = points.map(p => p[0]);
        const y = points.map(p => p[1]);
        const z = points.map(p => p[2]);

        const i = triangles.map(t => t[0]);
        const j = triangles.map(t => t[1]);
        const k = triangles.map(t => t[2]);

        const mesh = {
            type: 'mesh3d',
            x: x,
            y: y,
            z: z,
            i: i,
            j: j,
            k: k,
            intensity: z,
            colorscale: 'Viridis',
            opacity: 0.8
        };

        await Plotly.newPlot('plot', [mesh], {
            title: 'Delaunay Triangulation',
            autosize: true,
            scene: {
                xaxis: {title: 'X'},
                yaxis: {title: 'Y'},
                zaxis: {title: 'Elevation'}
            }
        });

        statusText.innerText = "✅ Побудова завершена успішно!";
        console.log("Графік побудовано успішно!");
    } catch (error) {
        statusText.innerText = "❌ Сталася помилка: " + error.message;
        console.error("Помилка:", error);
    } finally {
        setTimeout(() => {
            loader.style.display = "none";
        }, 1000);
    }
});
