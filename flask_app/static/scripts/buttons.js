window.addEventListener("load", function () {
  //your js here
  //console.log(document.getElementById("gpt3.5"));
  /* const entitiyDropdowns = document.getElementsByClassName(
    "dropdown-item entities"
  );
  for (let item of entitiyDropdowns) {
    item.addEventListener("click", bla);
  } */
});

function bla(e) {
  e.preventDefault();
  const method = e.target.id;
  const sorting = e.target.classList[1];
  var jsonData = jQuery_1_10_2.ajax({
    type: "POST",
    url: "/bio_viewer_sort",
    dataType: "json",
    contentType: "application/json",
    data: JSON.stringify({ method: method, sorting }),
    async: false,
  }).responseText;
  //var objectData = JSON.parse(jsonData);
}
