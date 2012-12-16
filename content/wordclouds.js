/*jslint browser: true*/
$(window).load(function () {
    //"use strict";
    var chartWidth = 600;
    var chartHeight = 200;
    // TODO don't hardcode from jsplotlib
    var AXIS_PROPORTION = .12;
    var sliderWidth = chartWidth * (1-AXIS_PROPORTION);
    var topHistogram;
    //////////////////////////////////
    // set up the slider and histogram
    $.ajaxSetup({ async: false });
    $.post('/getsliderbins', {}, function (data) {
        // set up the histogram
        var histogramChart = jsplotlib.make_chart(chartWidth,chartHeight,
            "#chart-area",":first-child",{"id":"total-histogram"});
        N = data.other.length;
        topHistogram = jsplotlib.bar_graph(histogramChart)
            .data([data.other.map(function(d) { return d / 1000; }),
                   data.me.map(function(d) { return d / 1000 ; })])
            //.data([[1,2,3,4,5],[9,7,5,3,1]]) // test data
            .series_labels(["other","me"])
            .xlabel("Date")
            .ylabel("Thousands of words")
            .xrange(new Date(data.mindate),new Date(data.maxdate), N)
            .xformat(d3.time.format("%B %Y"));
        topHistogram.draw();
        // set up the sliders: note that they only work in chrome :(
        var N = data.me.length;
        var maxValue = parseInt(data.me.length, 10) - 1;
        var actualSliderWidth = (sliderWidth/N*(N-1)).toString();
        $('#start-slider')
            .attr('max', maxValue)
            .attr('value', 0)
            .css('width', actualSliderWidth + "px")
            .css('padding-right', sliderWidth/N);

        $('#end-slider')
            .attr('max', maxValue)
            .attr('value', maxValue)
            .css( 'width', actualSliderWidth + "px")
            .css( 'padding-left',sliderWidth/N);

        $('#chart-area')
            .css('width', chartWidth + 'px')
            .css('margin', 'auto');

        $('.slider-container')
            .css('float', 'right');

    });
    $.ajaxSetup({async: true});

    // testing stuff
    var sizeColorCount, size, color, count,setOnePerson;

    setOnePerson = function (data, person) {
        var label;
        $('#' + person + '-cloud').html(data.textbody);
        for (label in data) {
            if (label.indexOf("wordcloud" + person) === 0) {
                sizeColorCount = data[label].split(',');
                size = sizeColorCount[0];
                color = sizeColorCount[1];
                count = sizeColorCount[2];
                $("#" + label).css("font-size", parseInt(size, 10));
                $("#" + label).css("color", "#" + color);
                $("#" + label).attr("title", count + " times");
            };
        };
    };
    var resetWords = function () {
        $.post('/computewords', {}, function (fulldata) {
            setOnePerson(fulldata.me, 'me');
            setOnePerson(fulldata.other, 'other');
        });
    };
    resetWords();
    ///////////////////////////////////
    // slider handling
    // TODO expose bars better in jsplotlib interface
    var updateChart = function (chart,start,end) {
        topHistogram._subbars
            .attr("is_active", function(d,i) { 
                return (i >= start) && (i <= end);
            });
    };
    var updateBounds = function () {
        var start = $("#start-slider").val();
        var end = $("#end-slider").val();
        updateChart(topHistogram,start,end);
        postdata = { start: start, end: end };
        $.post('/updatebounds',postdata, function(data) {
            $('#end-slider-label').html(data.end);
            $('#start-slider-label').html(data.start);
        });
        resetWords();
    };

    $("#start-slider").change(function() {
        if (parseInt($("#end-slider").val(),10) < parseInt(this.value,10) ){
            $("#end-slider").val(this.value);
        };
        updateBounds()
    });
    $("#end-slider").change(function() {
        if (parseInt($("#start-slider").val(),10) > parseInt(this.value,10) ){
            $("#start-slider").val(this.value);
        };
        updateBounds()
    });
    updateBounds()

    // TODO name handling
});
