# CSS styles for the HTML representation
_stylesheet = """

/* TENSORBLOCK STYLES*/
.tensorblock-header{
    padding: 5px 0;
    border: solid gray;
    border-width: 0 0 1px 0;
    font-weight: bold
}

.tensorblock-shape{
    font-weight: normal
}

.tensorblock-content{
    padding: 5px
}

/* STYLES FOR LABELS WITHIN TENSORBLOCKS */
.labels-container{
    display: flex;
    margin: 5px 10px;
    border: solid lightblue 3px;
    border-width: 0 0 0 3px;
}

.labels-names{
    padding: 0 10px; 
    margin: 0 10px 0 0; 
    background-color: aliceblue
}

.labels-values{
    flex: 1;
    display: flex;
    overflow-x: auto
}

/* The following three styles produce a scrollbar 
for label values that looks better than the default one */

.labels-values::-webkit-scrollbar {
    height: 4px;
}

.labels-values::-webkit-scrollbar-track {
    border-radius: 8px;
    background-color: #e7e7e7;
    border: 1px solid #cacaca;
    box-shadow: inset 0 0 6px rgba(0, 0, 0, .3);
}

.labels-values::-webkit-scrollbar-thumb {
    border-radius: 8px;
    background-color: #363636;
}

/* End of scrollbar stuff */

.labels-value{
    padding: 0 2px;
    text-align: center;
}

/* TENSORMAP STYLES */
.tensormap-container{
    background-color:white;
    width: 100%;
    padding: 0 0 10px 0
}

.tensormap-header{
    padding: 5px 0;
    border: solid gray;
    border-width: 0 0 1px 0;
    font-weight: bold
}

.tensormap-container .tensorblock-header{
    display: none
}

.tensormap-keyscontainer{
    padding:5px 10px
}

.tensormap-blockscontainer{
    padding:0 10px;
    box-sizing: border-box;
    width: 100%
}

.tensormap-blockslist{
    border: 3px solid black; 
    border-radius: 2px;
    margin: 0 0 0 10px;
    box-sizing: border-box;
    width: 100%
}

.tensormap-blockrow{
    border: 2px solid black; 
    border-width:0 0 2px 0;
    transition: 300ms;
    box-sizing: border-box;
    width: 100%
}

.tensormap-blockrow summary:hover{
    cursor: pointer;
}

.tensormap-blockrow.last{
    border-width: 0
}

.tensormap-blockrow.odd{
    background-color: white
}

.tensormap-blockrow.even{
    background-color: whitesmoke
}

.tensormap-blockrow.even:hover{
    background-color: gainsboro;
}

.tensormap-blockrow.odd:hover{
    background-color: gainsboro;
}

.tensormap-blockrow summary {
    display: flex
}

.tensormap-keysheader{
    border:2px black solid; 
    border-width:0 1px 0 0; 
    padding: 4px 20px
}

.tensormap-blockheader{
    padding:4px 20px
}

.tensormap-blockcollapsible{
    display: flex;
    border: solid black 1px;
    border-width: 1px 0 0 0;
    width: 100%
}

.tensormap-blockkeys{
    border: dashed black 1px;
    border-width: 0 1px 0 0;
    padding: 4px 20px;
}

.tensormap-blockrepr{
    padding: 4px 10px;
    flex: 1;
    overflow: hidden
}

"""