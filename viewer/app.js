let DATA;

fetch("../embedding_data/embeddings.json")
  .then(r => r.json())
  .then(json => {

    DATA = json;

    const camSel = document.getElementById("camera");

    json.datasets.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.dataset_id;
      opt.text = d.dataset_id;
      camSel.appendChild(opt);
    });

    for (let i = 0; i < 5; i++) {
      ["x", "y", "z"].forEach(id => {
        const opt = document.createElement("option");
        opt.value = i;
        opt.text = "PC" + (i + 1);
        document.getElementById(id).appendChild(opt);
      });
    }

    updatePlot();
  });

document
  .querySelectorAll("select")
  .forEach(el => el.addEventListener("change", updatePlot));

  function updatePlot() {

    const cam = camera.value;

    console.log(cam);
    const modeVal = mode.value;
    const xi = +x.value;
    const yi = +y.value;
    const zi = +z.value;
  
    const dataset =
      DATA.datasets.find(d => d.dataset_id === cam);
  
    const xLabel = "PC" + (xi + 1);
    const yLabel = "PC" + (yi + 1);
    const zLabel = "PC" + (zi + 1);
  
    let titleText;

    if (modeVal === "3d") {
        titleText = `<b>${cam}</b> – ${xLabel} vs ${yLabel} vs ${zLabel}`;
          document.getElementById("z-select").style.display = "block";
    }

    else {
        titleText = `${cam} - ${xLabel} vs ${yLabel}`;
        document.getElementById("z-select").style.display = "none";
    }
  
    const traces = [];
  
    Object.keys(dataset.label_map).forEach(labelKey => {

      const label = +labelKey;
    
      const pts = dataset.points.filter(
        p => p.label === label
      );
    
      if (pts.length === 0) return;
    
      const trace = {
        x: pts.map(p => p.pcs[xi]),
        y: pts.map(p => p.pcs[yi]),
        z: pts.map(p => p.pcs[zi]),
      
        mode: "markers",
        type: modeVal === "3d" ? "scatter3d" : "scatter",
    
        marker: {
          size: modeVal === "3d" ? 6 : 10,
          opacity: 0.8,
          color: dataset.label_map[label].color
        },
      
        name: dataset.label_map[label].name,
      
        customdata: pts.map(p => p.image_url),

        hoverinfo: "none"
          
      };      
    
      traces.push(trace);
    });

  
    const layout = {
      title: { text: titleText },
      showlegend: true,
    
      legend: {
        itemsizing: "constant",
        font: { size: 12 }
      }
    };
    
    if (modeVal === "3d") {
      layout.scene = {
        xaxis: { title: {text: xLabel} },
        yaxis: { title: {text: yLabel} },
        zaxis: { title: {text: zLabel} }
      };
    } else {
      layout.xaxis = { title: {text: xLabel} };
      layout.yaxis = { title: {text: yLabel} };
    }

  Plotly.purge("plot");

  Plotly.newPlot("plot", traces, layout).then(() => {

    const plotDiv = document.getElementById("plot");
    const preview = document.getElementById("img-preview");
    const previewImg = document.getElementById("preview-img");

    preview.style.position = "absolute";
    preview.style.pointerEvents = "none";
    preview.style.display = "none";
    preview.style.opacity = 0;

    let hoverTimeout = null;
    let lastPointId = null;
    let mouseX = 0;
    let mouseY = 0;

    plotDiv.addEventListener("mousemove", evt => {
      mouseX = evt.pageX;
      mouseY = evt.pageY;
    });


    if (modeVal === "2d") {

      plotDiv.on("plotly_hover", data => {

        const pt = data.points[0];

        previewImg.src = pt.customdata;
        preview.style.display = "block";

        preview.style.left = mouseX + 20 + "px";
        preview.style.top  = mouseY + 20 + "px";

        preview.style.opacity = 1;
      });

      plotDiv.on("plotly_unhover", () => {
        preview.style.display = "none";
        preview.style.opacity = 0;
      });
    }


    if (modeVal === "3d") {

      plotDiv.on("plotly_hover", data => {

        const pt = data.points[0];

        if (pt.pointNumber === lastPointId) return;
        lastPointId = pt.pointNumber;

        clearTimeout(hoverTimeout);

        hoverTimeout = setTimeout(() => {

          previewImg.src = pt.customdata;
          preview.style.display = "block";

          preview.style.left = mouseX + 20 + "px";
          preview.style.top  = mouseY + 20 + "px";

          preview.style.transition = "opacity 0.2s";
          preview.style.opacity = 1;

        }, 800); 
      });

      plotDiv.on("plotly_unhover", () => {

        lastPointId = null;
        clearTimeout(hoverTimeout);

        preview.style.opacity = 0;

        setTimeout(() => {
          preview.style.display = "none";
        }, 200);
      });
    }
  });
}