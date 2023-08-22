window.addEventListener("load", function () {
  //your js here
  //console.log(document.getElementById("gpt3.5"));
  const dropdowns = document.getElementsByClassName("dropdown-item");
  console.log(dropdowns);
  for (let item of dropdowns) {
    item.addEventListener("click", bla);
  }
});

function bla(e) {
  e.preventDefault();
  const method = e.target.id;
  var jsonData = jQuery_1_10_2.ajax({
    type: "POST",
    url: "/bio_viewer",
    dataType: "json",
    contentType: "application/json",
    data: JSON.stringify({ method: method }),
    async: false,
  }).responseText;
  //var objectData = JSON.parse(jsonData);
}
