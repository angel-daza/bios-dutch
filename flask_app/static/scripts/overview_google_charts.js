// Set a callback to run when the Google Visualization API is loaded.
google.charts.setOnLoadCallback(drawBioOverviewCharts);

var level = 0;

function drawBioOverviewCharts() {
  const chartWidth = 500;
  const chartHeight = 300;

  // Just make ONE Call to the Server that Brings The whole Biography JSON Content
  var jsonData = jQuery_1_10_2.ajax({
    url: "/bio_overview_google_charts",
    dataType: "json",
    async: false,
  }).responseText;
  var objectData = JSON.parse(jsonData);

  // ##########################################  PIE CHART  ##########################################
  // Create the Entity Frequency data table.
  var methodData = new google.visualization.arrayToDataTable(
    objectData["method_total"]
  );
  // Set chart options
  var methodOptions = {
    title: objectData["options"].title,
    pieSliceText: objectData["options"].pieSliceText,
    width: chartWidth,
    height: chartHeight,
  };
  // Instantiate and draw our chart, passing in some options.
  var methodChart = new google.visualization.PieChart(
    document.getElementById("chart_method_total")
  );

  methodChart.draw(methodData, methodOptions);

  var categoryData = new google.visualization.arrayToDataTable(
    objectData["category_total"]
  );
  // Set chart options
  var categoryOptions = {
    title: objectData["options"].categoryTitle,
    pieSliceText: objectData["options"].pieSliceText,
    width: chartWidth,
    height: chartHeight,
  };
  // Instantiate and draw our chart, passing in some options.
  var categoryChart = new google.visualization.PieChart(
    document.getElementById("chart_category_total")
  );

  methodChart.draw(methodData, methodOptions);
  categoryChart.draw(categoryData, categoryOptions);

  const categoryMethodData = objectData["per_method_category"];

  google.visualization.events.addListener(methodChart, "select", selectHandler);

  function selectHandler() {
    if (level === 0) {
      var selectedItem = methodChart.getSelection()[0];
      if (selectedItem) {
        const index = selectedItem.row + 1;
        const newDKey = objectData["method_total"][index][0];
        const newRow = objectData["per_method_category"][newDKey];
        const newOptions = { ...methodOptions };
        newOptions.title = newRow[0][0];
        //console.log(objectData["method_total"][selectedItem.row + 1]);
        //var newValue = methodData.getValue(selectedItem.row, 1);
        updateChart(newRow, newOptions);
        level = 1;
      }
    } else {
      updateChart(objectData["method_total"], methodOptions);
      level = 0;
    }
  }

  function updateChart(newDataArray, newOptions) {
    // Generate new data based on the selectedValue
    var newData = google.visualization.arrayToDataTable(newDataArray);

    // Update the chart with new data
    methodChart.draw(newData, newOptions);
  }
}
