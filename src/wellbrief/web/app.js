// Placeholder page script: fetches this server's own /api/status and renders it. The real
// ask/brief/npt views are a later step; this only proves the static/API wiring and sets the
// pattern later scripts on this page must follow: build the DOM with textContent, never assign
// markup built from data this page did not itself write, and only ever address this same
// origin, no other scheme or host.
"use strict";

function setText(id, value) {
  var el = document.getElementById(id);
  if (el) {
    el.textContent = value;
  }
}

function loadStatus() {
  fetch("/api/status")
    .then(function (response) {
      return response.json();
    })
    .then(function (status) {
      setText("status-workspace", status.workspace);
      setText("status-network-mode", status.network_mode);
      setText("status-blocked", String(status.outbound_connection_attempts));
    })
    .catch(function () {
      setText("subtitle", "could not reach the local API");
    });
}

loadStatus();
