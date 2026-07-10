document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  if (dropzone && fileInput) {
    ["dragenter", "dragover"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      })
    );
    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length) {
        fileInput.files = files;
      }
    });
  }

  const radios = document.querySelectorAll('input[name="verdict"]');
  const actualSelect = document.getElementById("actual_label");
  if (radios.length && actualSelect) {
    const sync = () => {
      const checked = document.querySelector('input[name="verdict"]:checked');
      const incorrect = checked && checked.value === "incorrect";
      actualSelect.disabled = !incorrect;
      actualSelect.required = !!incorrect;
      if (!incorrect) actualSelect.value = "";
    };
    radios.forEach((r) => r.addEventListener("change", sync));
    sync();
  }
});
