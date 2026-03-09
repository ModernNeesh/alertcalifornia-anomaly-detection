let DATA;

fetch("../embedding_data/embeddings.json")
  .then(r => r.json())
  .then(json => {

    DATA = json;

    // dataset selection
    const datasetSel = document.getElementById("dataset");

    json.datasets.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.dataset_id;
      opt.text = d.dataset_id;
      datasetSel.appendChild(opt);
    });
    
    // pc selection
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

document.querySelectorAll("select").forEach(el => el.addEventListener("change", updatePlot));

  // collect all unique values across datasets
  let values = new Set();

function updateColorOptions(selectedDatasets) {

  const colorBySelect = document.getElementById("colorBy");

  // check if selected dataset contains camera_locations
  const hasLocations = selectedDatasets.some(ds =>
    ds.points.some(p => p.hasOwnProperty("camera_locations"))
  );

  const existingCamLoc = colorBySelect.querySelector(
    'option[value="camera_locations"]'
  );

  // add option if camera_locations exists in any selected dataset and option not already present
  if (hasLocations && !existingCamLoc) {
    const opt = document.createElement("option");
    opt.value = "camera_locations";
    opt.text = " Location";
    colorBySelect.appendChild(opt);
  }
  
  // remove option if no selected datasets have camera_locations
  if (!hasLocations && existingCamLoc) {
    if (colorBySelect.value === "camera_locations") {
      colorBySelect.value = "label"; 
    }
    existingCamLoc.remove();
  }

  // check if multiple datasets are selected
  const existingDataset = colorBySelect.querySelector(
    'option[value="dataset"]'
  );

  const multipleDatasets = selectedDatasets.length > 1;

  // add camera option if multiple datasets selected and option not already present
  if (multipleDatasets && !existingDataset) {
    const opt = document.createElement("option");
    opt.value = "dataset";
    opt.text = "Dataset";
    colorBySelect.appendChild(opt);
  }

  // remove camera option if not multiple datasets and option currently present
  if (!multipleDatasets && existingDataset) {

    if (colorBySelect.value === "dataset") {
      colorBySelect.value = "label";
    }

    existingDataset.remove();
  }
}

function updatePlot() {

  // get selected cameras 

  const datasetSelect = document.getElementById("dataset");
  const selectedSets =
    [...datasetSelect.selectedOptions].map(o => o.value);

  if (selectedSets.length === 0) {
    Plotly.purge("plot");
    return;
  }

  const selectedDatasets = DATA.datasets.filter(d =>
    selectedSets.includes(d.dataset_id)
  );


  updateColorOptions(selectedDatasets);

  let allPoints = [];

  selectedDatasets.forEach(ds => {
    ds.points.forEach(p => {
      allPoints.push({
        ...p,
        dataset_id: ds.dataset_id,
        label_map: ds.label_map
      });
    });
  });

  // map label names
  const labelMap = selectedDatasets[0].label_map;
  allPoints.forEach(p => {
    p.label_name = labelMap[p.label]?.name ?? p.label;
  });

  modeVal = mode.value;
  const xi = +document.getElementById("x").value;
  const yi = +document.getElementById("y").value;
  const zi = +document.getElementById("z").value;

  const traces = [];

  // define color scale for labels
  const colorMode = colorBy.value;
  if (colorMode === "label") {
    const labelMap = selectedDatasets[0].label_map;
  
    allPoints.forEach(p => {
      const labelInfo = labelMap[p.label];
      p.label_name = labelInfo.name;
      p.color = labelInfo.color;
    });
  }
  else if (colorMode === "dataset") {
    const camSet = new Set(allPoints.map(p => p.dataset_id));
    const cams = Array.from(camSet).sort();

    console.log(cams)
    const colors = ["#b9a8d5", "#96b9d0", "#73b587", "#ffd852", "#fa9b1a", "#ead1db", "#c4e1e5"];

    const colorMap = {};

    cams.forEach((cam, i) => {
      colorMap[cam] = colors[i % colors.length];
    });

    allPoints.forEach(p => {
      p.color = colorMap[p.dataset_id];
    });

  }

  else if (colorMode === "season") {
    allPoints.forEach(p => {
      const season = p.season;
      if (season === "Spring") p.color = "#009e74";
      else if (season === "Summer") p.color = "#cd79a8";
      else if (season === "Fall") p.color = "#e79e00";
      else if (season === "Winter") p.color = "#89ccef";
    });
  }

  else if (colorMode === "hour") {
    allPoints.forEach(p => {
      const hour = p.hour;
      if (hour >= 5 && hour < 12) p.color = "#009e74";
      else if (hour >= 12 && hour < 17) p.color = "#e79e00"; 
      else if (hour >= 17 && hour < 21) p.color = "#cd79a8"; 
      else p.color = "#89ccef"; 
    });
  }

  else if (colorMode === "daytime") {
    allPoints.forEach(p => {
      const isDay = p.daytime;
      p.color = isDay ? "orange" : "#03629c";
    });
  }

  else if (colorMode === "camera_locations") {

    const locationSet = new Set(allPoints.map(p => p.camera_locations));
    const locations = Array.from(locationSet).sort();
  
    const colors = ["#6e963b", "#f8cf05", "#172c49"];
  
    const colorMap = {};
  
    locations.forEach((loc, i) => {
      colorMap[loc] = colors[i % colors.length];
    });
  
    allPoints.forEach(p => {
      p.color = colorMap[p.camera_locations] || "#cccccc";
    });
  }

  const groups = {};

  allPoints.forEach(p => {

    let groupKey;

    if (colorMode === "label") groupKey = p.label_name;
    else if (colorMode === "dataset") groupKey = p.dataset_id;
    else if (colorMode === "season") groupKey = p.season;
    else if (colorMode === "hour") {
      const hour = p.hour;
      if (hour >= 5 && hour < 12) groupKey = "Morning";
      else if (hour >= 12 && hour < 17) groupKey = "Afternoon"; 
      else if (hour >= 17 && hour < 21) groupKey = "Evening"; 
      else groupKey = "Night";
    }
    else if (colorMode === "daytime") groupKey = p.daytime ? "Day" : "Night";
    else if (colorMode === "camera_locations") groupKey = p.camera_locations;
    else groupKey = "All";

    if (!groups[groupKey]) groups[groupKey] = [];
    groups[groupKey].push(p);
  });

  Object.keys(groups).forEach(key => {

    const pts = groups[key];
  
    traces.push({
      x: pts.map(p => p.pcs[xi]),
      y: pts.map(p => p.pcs[yi]),
      z: modeVal === "3d" ? pts.map(p => p.pcs[zi]) : undefined,
      type: modeVal === "3d" ? "scatter3d" : "scatter",
      mode: "markers",
      name: key,
      marker: {
        size: modeVal === "3d" ? 6 : 12,
        color: pts.map(p => p.color), 
        opacity: 0.55
      },
      customdata: pts.map(p => [
        p.img_url,
        key,
        p.color,
        p.id,
        p.label_name
      ]),
      hoverinfo: "none"
      });
  });

  // define layout 
  const xLabel = "PC" + (xi + 1);
  const yLabel = "PC" + (yi + 1);
  const zLabel = "PC" + (zi + 1);

  const layout = {
    font: {
      family: "Roboto, sans-serif",
    },
    title: {
      font: { size: 30 },
      text: selectedSets.join(", ") 
      // + 
      //   (modeVal === "3d" 
      //   ? `<br><span style="font-size:40px">${xLabel} vs ${yLabel} vs ${zLabel}</span>` 
      //   : `<br><span style="font-size:40px">${xLabel} vs ${yLabel}</span>`)
    },
    showlegend: true,
    legend: {
      itemsizing: "constant",
      font: { size: 30 }
    }
  };

  if (modeVal === "3d") {
    layout.scene = {
      xaxis: { title: { text: xLabel, font: { size: 24 } }, tickfont: { size: 14 } },
      yaxis: { title: {text: yLabel, font: { size: 24 } }, tickfont: { size: 14 } },
      zaxis: { title: {text: zLabel, font: { size: 24 } }, tickfont: { size: 14 } },
      aspectmode: "cube"
    },
    document.getElementById("z-select").style.display = "block";

  } else {
    layout.xaxis = { title: {text: xLabel, font: { size: 24 } },
    scaleanchor: "y",
    scaleratio: 1,
    tickfont: { size: 22 },     
  };
    layout.yaxis = { title: {text: yLabel, font: { size: 24 } },
    tickfont: { size: 22 } };
    document.getElementById("z-select").style.display = "none";
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
    
        const imgUrl = pt.customdata[0];
        const colorLabel = pt.customdata[1];
        const labelColor = pt.customdata[2];
        const id = pt.customdata[3];
        classLabel = pt.customdata[4];
    
        previewImg.src = imgUrl;
    
        previewImg.style.border = `3px solid ${labelColor}`;
    
        const labelDiv = document.getElementById("preview-label");
        if (labelDiv) {
          const colorByVal = document.getElementById("colorBy").value;
          const colorLine = colorByVal !== "label"
            ? `<span class="preview-color-group">${colorLabel}</span>`
            : "";

          labelDiv.innerHTML = `
            <span class="preview-class">${classLabel}</span>
            ${colorLine}
            <span class="preview-id">ID: ${id}</span>
          `;
        }
    
        preview.style.display = "block";
        preview.style.left = mouseX + 10 + "px";
        preview.style.top  = mouseY + 10 + "px";
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
    
        const imgUrl = pt.customdata[0];
        const colorLabel = pt.customdata[1];
        const labelColor = pt.customdata[2];
        const id = pt.customdata[3];
        classLabel = pt.customdata[4];
    
        previewImg.src = imgUrl;
    
        previewImg.style.border = `3px solid ${labelColor}`;
    
        const labelDiv = document.getElementById("preview-label");
        if (labelDiv) {
          const colorByVal = document.getElementById("colorBy").value;
          const colorLine = colorByVal !== "label"
            ? `<span class="preview-color-group">${colorLabel}</span>`
            : "";

          labelDiv.innerHTML = `
            <span class="preview-class">${classLabel}</span>
            ${colorLine}
            <span class="preview-id">ID: ${id}</span>
          `;
        }
    
          preview.style.display = "block";
          preview.style.left = mouseX + 10 + "px";
          preview.style.top  = mouseY + 10 + "px";
    
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