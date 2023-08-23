// Set a callback to run when the Google Visualization API is loaded.
google.charts.setOnLoadCallback(drawBioStatCharts);

// Callback that creates and populates a data table,
// instantiates the pie chart, passes in the data and
// draws it.
function drawBioStatCharts() {
  const box = document.getElementById("bioId");

  // Just make ONE Call to the Server that Brings The whole Biography JSON Content
  var jsonData = jQuery_1_10_2.ajax({
    url: "/bio_ids/",
    dataType: "json",
    async: false,
  }).responseText;
  var objectData = JSON.parse(jsonData);
  const ids = objectData["ids"];

  // ##########################################  PIE CHART  ##########################################
  // Create the Entity Frequency data table.
  var data = new google.visualization.DataTable(
    JSON.stringify(objectData["entity_freq_table"])
  );
  // Set chart options
  var options = {
    title: objectData["entity_freq_title"],
    width: 600,
    height: 400,
  };
  // Instantiate and draw our chart, passing in some options.
  var chart = new google.visualization.PieChart(
    document.getElementById("chart_tot_entities")
  );
  chart.draw(data, options);

  // ##########################################  [STACKED?] BAR CHART  ##########################################

  var data_bar = google.visualization.arrayToDataTable(
    objectData["model_entity_dist"]
  );

  var options_bar = {
    width: 600,
    height: 400,
    legend: { position: "top", maxLines: 3 },
    bar: { groupWidth: "75%" },
    // isStacked: true
  };

  // Instantiate and draw our chart, passing in some options.
  var chart_bar = new google.visualization.BarChart(
    document.getElementById("chart_all_models_count")
  );
  chart_bar.draw(data_bar, options_bar);
}
