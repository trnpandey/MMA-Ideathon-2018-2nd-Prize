var express=require('express');
var app=express();
var bodyParser=require('body-parser');
var request=require('request');

var err="No information available: Please check the details that you have provided";

app.use(bodyParser.json());
app.use(express.static(__dirname+'/public'));

var port = process.env.PORT || 5000;
	

var server=app.listen(process.env.PORT || 5000, function () {
    var port = server.address().port;
    console.log("App now running on port", port);
  });
  


app.get('/',function(req,res){
	res.status(200).send('Welcome to Ami\'s RESTFUL Server');
});

//Custom Search Engine
app.get('/customsearchengine/:query',function(req,res){
	

	
	var query=req.params.query;
	
	var result="";
	
	function second()
	{
		return res.status(200).send(JSON.parse(result));
	}
	
	function first(callback)
	{
		
		var str,find;
		
		request('https://www.googleapis.com/customsearch/v1?key=API_KEY&cx=Engine_ID&q='+'buy '+query+' online', function (error, response, body) {
			if (!error && response.statusCode == 200) 
			{
			var o=JSON.parse(body);
			result=result.concat("[");
			
			for(var i=0;i<o.items.length;i++)
			{
					result=result.concat("{");
					
					str=o.items[i].title
					find = '\n';
					var re = new RegExp(find, 'g');
					str = str.replace(re, '');
					result=result.concat("\"title\":\"").concat(str).concat("\",");
					
					result=result.concat("\"link\":\"").concat(o.items[i].link).concat("\",");
					result=result.concat("\"displaylink\":\"").concat(o.items[i].displayLink).concat("\",");
					
					str=o.items[i].snippet;
					find = '\n';
					var re = new RegExp(find, 'g');
					str = str.replace(re, '');
					result=result.concat("\"description\":\"").concat(str).concat("\",");
					
					if(o.items[i].pagemap.cse_thumbnail == null)
					result=result.concat("\"image\":\"").concat("https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg").concat("\"");
					else
					result=result.concat("\"image\":\"").concat(o.items[i].pagemap.cse_thumbnail[0].src).concat("\"");
				
				
			
					
					result=result.concat("}");
					if(i<=o.items.length-2)
					result=result.concat(",");
				
				
			}
			
			result=result.concat("]");
				
			callback(second);
			}
		});
		
		
	}
	
	
	
	
	first(second);
	

});






