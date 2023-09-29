function setAction(){
    const path = "/create";
    const key = "app_name";
    const val = document.getElementById('name-form').value;
    document.getElementById('upload-form').action = path+'?'+ key +'='+ val;
}