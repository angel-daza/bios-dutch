window.addEventListener("load", function () {
  google.charts.setOnLoadCallback(drawBioStatCharts);
});

// Callback that creates and populates a data table,
// instantiates the pie chart, passes in the data and
// draws it.
function drawBioStatCharts() {
  // Just make ONE Call to the Server that Brings The whole Biography JSON Content
  var jsonData = jQuery_1_10_2.ajax({
    url: "/bio_stat_google_charts",
    dataType: "json",
    async: false,
  }).responseText;
  var objectData = JSON.parse(jsonData);
  var arrayData = objectData["distance_array"];

  for (const [key, value] of Object.entries(arrayData)) {
    const boxID = "stat_total_" + key;
    if (document.getElementById(boxID)) {
      //console.log(key, value);
      const dataTable = value;
      var data = new google.visualization.arrayToDataTable(dataTable);
      var options = {
        title: "Total #Entities",
        width: 400,
        height: 200,
        legend: { position: "none" },
      };
      // Instantiate and draw our chart, passing in some options.
      var chart = new google.visualization.BarChart(
        document.getElementById(boxID)
      );
      chart.draw(data, options);
    }
  }

  var devArrayData = objectData["dev_array"];

  for (const [key, value] of Object.entries(devArrayData)) {
    const devBoxID = "stat_dev_" + key;
    if (document.getElementById(devBoxID)) {
      //console.log(key, value);
      const devDataTable = value;
      var devData = new google.visualization.arrayToDataTable(devDataTable);
      var devOptions = {
        title: "Gold Deviation",
        width: 400,
        height: 200,
        legend: { position: "none" },
        bar: { groupWidth: "75%" },
        //isStacked: true,
        legend: { position: "top", maxLines: 3 },
      };
      // Instantiate and draw our chart, passing in some options.
      var devChart = new google.visualization.BarChart(
        document.getElementById(devBoxID)
      );
      devChart.draw(devData, devOptions);
    }
  }
}
