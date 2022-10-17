function(token_val, width, height, size, model_choice,a,b){
    let app=document.querySelector("gradio-app").shadowRoot;
    app.querySelector("#sdinfframe").style.height=height+"px";
    let frame=app.querySelector("#sdinfframe").contentWindow.document;
    if(frame.querySelector("#setup").value=="0")
    {
        window.my_setup=setInterval(function(){
            let frame=document.querySelector("gradio-app").shadowRoot.querySelector("#sdinfframe").contentWindow.document;
            console.log("Check PyScript...")
            if(frame.querySelector("#setup").value=="1")
            {
                frame.querySelector("#draw").click();
                clearInterval(window.my_setup);
            }
        },100)
    }
    else
    {
        frame.querySelector("#draw").click();
    }

    if(!window.my_observe_upload)
    {
        console.log("setup upload here");
        window.my_observe_upload = new MutationObserver(function (event) {
            console.log(event);
            var frame=document.querySelector("gradio-app").shadowRoot.querySelector("#sdinfframe").contentWindow.document;
            frame.querySelector("#upload").click();
        });
        window.my_observe_upload_target = document.querySelector("gradio-app").shadowRoot.querySelector("#upload span");
        window.my_observe_upload.observe(window.my_observe_upload_target, {
            attributes: false, 
            subtree: true,
            childList: true, 
            characterData: true
        });
    }
    return [token_val, width, height, size, model_choice,a,b];
}