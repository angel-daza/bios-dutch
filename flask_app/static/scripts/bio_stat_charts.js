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
  var arrayData = objectData["array_of_dicts"];

  for (const [key, value] of Object.entries(arrayData)) {
    const boxID = "stat_" + key;
    if (document.getElementById(boxID)) {
      //console.log(key, value);
      const dataTable = value;
      var data = new google.visualization.arrayToDataTable(dataTable);
      var options = {
        title: "#Entities found",
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
}
