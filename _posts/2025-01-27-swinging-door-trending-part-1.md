---
layout: post
title: Swinging Door Trending
tags: 
- Algorithms
- Python
- SCADA 
---

Time-series data may be sampled at a constant rate, such is the case with polling, by some event at irregular intervals, such as when a change occurs. When the value being measured is changing rapidly or sampled at a very fast rate, the quantity of samples being accumulated could become very large. This presents challenges for retention of time-series data within databases, as rapidly sampled data requires greater storage space. It is often the case that polling at a rate which is much faster than the process being measured provides no additional useful information, or that small fluctuations in the time-series data are of no interest. Time-series compression algorithms can be applied to reduce the quantity of samples in exchange for loss of some detail.  

At one point a number of years ago I was tasked with implementing a time series compression algorithm in Python based on the swinging door trending algorithm [[1]](#1). Swinging door trending (SDT) is a time-series compression algorithm which reduces a signal to a number of samples such that the deviation between the linear interpolation of the new samples and any of the original samples remains within a predetermined limit. In this post I will explain the SDT algorithm and provide an example in Python created to illustrate how the algorithm works. In a future post I will further cover the implementation as a proxy server with an Influx database and look at testing in a containerised environment with Docker and WireShark.

## Swing Door Trending Algorithm

The SDT algorithm can be described geometrically as an algorithm which attempts to enclose time-series data points within parallelogram envelopes constructed about a linear trend. As points are presented, the algorithm attempts to accomodate them within the constructed envelope. When a data point is presented that can not be accomodated within the current envelope, the algorithm generates a point marking the boundary between the current envelope and a new envelope which is created. The points at the extents of each linear segment forming the center line of a parallelogram are the data points in the compressed signal (Figure 1).

![Output points form a linear trend where each line segment is the longitudinal center line of a parallelogram envelope.](/assets/posts/swinging-door-trending-part-1/sdt1.png?raw=true){: width="50%"}
{:.image-caption}
*Figure 1: The SDT algorithm.*

At run-time, several states are retained by the algorithm in order to track the boundaries of the current envelope. The algorithm is initialised when the first value ($$FV1$$) is received. Referring to the figure below two points are calculated; the upper pivot ($$p_u$$) and lower pivot ($$p_l$$) which are a distance above and below the first value. This distance is a parameters of the algorithm refered to as the compression deviation. The line formed between these two pivot points is the left side of the first parallelogram. Two gradients, the sloping upper maximum ($$m_{u,max}$$) and sloping upper minimum ($$m_{l,min}$$) are initialised to infinity. These gradients, together with the upper and lower pivot, track the top and bottom edges of the envelope as it is being formed.

![First point determines the middle of the left-hand-side edge for the parallelogram envelope.](/assets/posts/swinging-door-trending-part-1/sdt2.png?raw=true){: width="50%"}
{:.image-caption}
*Figure 2: Initial state of the algorithm after receiving the first point.*

Once a second point is received, gradients are calculated between the upper pivot ($$p_u$$) and lower pivot ($$p_l$$) to the new point ($$p_2$$). These gradients are compared with sloping upper maximum ($$m_{u,max}$$) and sloping upper minimum ($$m_{l,min}$$) gradients, and the top and bottom sides of the envelope are widened to accommodate the new point (Figure 3).

![Top and bottom edges of parallelogram are widened to accomodae the new point.](/assets/posts/swinging-door-trending-part-1/sdt3.png?raw=true){: width="50%"}
{:.image-caption}
*Figure 3: Gradients adjusted to accomodate next point.*

Referring to the figure below, the process continues as more points are received and the envelope defined by $$m_{u,max}$$ and $$m_{l,min}$$ is further widened as required to accomodate these points. The effect of the gradients being increased as new points are presented is likened to swinging doors opening which is the basis for the naming of the algorithm.

![Top and bottom edges of parallelogram are continue to be widened to accomodae new points.](/assets/posts/swinging-door-trending-part-1/sdt4.png?raw=true){: width="50%"}
{:.image-caption}
*Figure 4: Gradients adjusted to accomodate further points.*

At a certain point, a measurement may be presented which would cause the right-hand-side of the envelope to exceed the compression deviation limit. This occurs when the recalculated $$m_{u,max}$$ and $$m_{l,min}$$ gradients are diverging away from each other (Figure 5).  

![New point causes the top and bottom edges of the envelope to diverge.](/assets/posts/swinging-door-trending-part-1/sdt5.png?raw=true){: width="50%"}
{:.image-caption}
*Figure 5: Point received which causes top and bottom edges of envelope to be diverging.*

Under this condition, a new output point will be generated by the algorithm. The output point is calculated by first setting the gradient which caused the divergence to be the same as the opposite gradient, resulting in parallel top and bottom edges for the parallelogram. The point of intersection between this gradient and the line between the latest and previous input points is found, which in this example is the lower right corner of the parallelogram (Figure 6). The centre of the right-hand side of the parallelogram is then calculated which is a new output point generated by the algorithm and the new first value used to form a new parallelogram envelope ($$FV_2$$). 

![Remaining points forming corners of parallelogram are found.](/assets/posts/swinging-door-trending-part-1/sdt6.png?raw=true){: width="50%"}
{:.image-caption}
*Figure 6: Point at the centre of the right hand side of the parallelogram is found.*

Using the new first value point ($$FV_2$$), new upper and lower pivot points are calculated ($$p_{u,2}$$ and $$p_{l,2}$$). The gradients $$m_{u,max}$$ and $$m_{l,min}$$ are recalculated as the gradients between the latest input point and the new upper and lower pivots. As new input points are received, the envelope is expanded until the gradients diverge and the compression deviation limit is exceeded, upon which the process repeats continues by returning a new output point which is the new first value (Figure 7). 

![Process is repeated to build further parallelogram envelopes.](/assets/posts/swinging-door-trending-part-1/sdt7.png?raw=true){: width="50%"}
{:.image-caption}
*Figure 7: The process repeats building futher parallelogram envelopes to suit new points as they are received.*

## Python Implementation

The following implementation of the SDT algorithm encapsulates the state and functionality of the algorithm in the *SdtFilter* class. The listing below shows the constructor method of this class. On initialisation, the $$m_{u,max}$$ (*_slopingUpperMax*) and $$m_{l,min}$$ (*_slopingLowerMin*) angles are set to a maximum value, as described in the previous section. 

{% highlight python %}
class SdtFilter(SerialFilter):

    def __init__(self, compressionDeviation, maxInterval):
        """ Class constructor. """

        # The compression deviation and maximum interval
        self._compressionDeviation = compressionDeviation
        self._maxInterval = maxInterval

        # Limits on sloping upper and sloping lower gradients
        self._slopingUpperMax = -float("inf")
        self._slopingLowerMin = float("inf")

        # Queu to hold up to two previous points
        self._lastPoints = deque(maxlen = 2)

        return

{% endhighlight %}

The *filterPoint()* method shown below is called to evaluate a new input point to the algorithm. When the function is called for the first time, the upper (*_upper*) and lower (*_lower*) pivot points are initialised and the point returned to the caller. On subsequent calls to *filterPoint()* the *_evaluateParallelogram()* method is called which updates the bounds of the parallelogram envelope and generates new output points when the envelope is exceeded. The *filterPoint()* method also evaluates the maximum time interval criteria. If the maximum time interval has been exceeded the *_updateWindow()* method is called to reset the paralleogram envelope and returns the last point to the caller as an output of the algorithm.   

{% highlight python %}

    def filterPoint(self, time, value) -> list:
        """ Applies compression to the time-series points. """
        results = list()

        # The current point
        thisPoint = FilterPoint(time = time, value = value)

        # Initialisation
        if(len(self._lastPoints) < 1):
            # Upper and lower pivot points
            self._upperPivot = thisPoint + FilterPoint(0, self._compressionDeviation)
            self._lowerPivot = thisPoint + FilterPoint(0, -self._compressionDeviation)
            
            # First point received is generated by the algorithm
            results += [(thisPoint.time, thisPoint.value)]
            self._firstTime = thisPoint.time
        # Handle invalid conditions
        elif(time <= self._lastPoints[0].time):
            raise ValueError("Time-series data-point must be newer than previous points.")
        else:
            # If maximum interval reached
            if((thisPoint.time - self._firstTime) > self._maxInterval):
                results += [(self._lastPoints[0].time, self._lastPoints[0].value)]
                # Recalculate the window
                self._updateWindow(self._lastPoints[0], thisPoint)  
                self._firstTime = self._lastPoints[0].time
            # If maximum interval still exceeded
            if((thisPoint.time - self._firstTime) > self._maxInterval):
                results += [(thisPoint.time, thisPoint.value)]
                # Recalculate the window
                self._updateWindow(self._lastPoints[0], thisPoint)
                self._firstTime = thisPoint.time  
                
            # Otherwise evaluate if parallelogram envelope exceeded
            else:
                results += self._evaluateParallelogram(thisPoint)

        # Save the last point
        self._lastPoints.appendleft(thisPoint)

        return results
{% endhighlight %}

The *_evaluateParallelogram()* algorithm below updates the bounds of the current parallelogram envelope. The the *slopingUpper* and *slopingLower* variables are gradients calculated between the latest input point and the upper and lower pivot points. If the *_sloperUpperMax* value is exceeded by *_slopingUpper* it is updated, and likewise for *_slopingLowerMin*. This causes the envelope to be widened to accomodate the latest input point. If the gradients are diverging, as indicated by the condition *_slopingUpperMax > _slopingLowerMin*, the gradient which was updated takes the value of the opposite gradient to form a parallelogram with parallel sides. A new output point is then calculated and the current envelope reset by through a call to the *_updateWindow()* method. The method then recursively calls itself to accomodate the latest input point.

{% highlight python %}
    def _evaluateParallelogram(self, thisPoint):
        results = list()

        # Update the sloping upper and sloping lower gradients
        # These are the gradients between the current point and upper/lower pivot points respectively
        slopingUpper = (thisPoint.value - self._upperPivot.value) / (thisPoint.time - self._upperPivot.time)
        slopingLower = (thisPoint.value - self._lowerPivot.value) / (thisPoint.time - self._lowerPivot.time)

        # If sloping upper gradient exceeded limit
        slopingUpperMaxUpdated = slopingUpper > self._slopingUpperMax 
        if(slopingUpperMaxUpdated):
            # Update sloping upper gradient limit
            self._slopingUpperMax = slopingUpper

        # If sloping lower gradient exceeded limit
        slopingLowerMinUpdated = slopingLower < self._slopingLowerMin
        if(slopingLowerMinUpdated):
            # Update sloping lower gradient minimum
            self._slopingLowerMin = slopingLower

        # If sloping upper gradient limit exceeds sloping lower gradient limit
        if(self._slopingUpperMax > self._slopingLowerMin):

            # If the upper gradient limit was exceeded L1 will be a line parallel 
            # to _slopingLowerMin, passing through the upper pivot
            if(slopingUpperMaxUpdated):
                # Use sloping lower min as the gradient
                m1 = self._slopingLowerMin
                # Find intercept from equation of line passing through upper pivot: 
                #   b1 = y - m1*x
                b1 = self._upperPivot.value - m1*self._upperPivot.time 
            # If the upper gradient limit was exceeded L1 will be a line parallel 
            # to _slopingUpperMax, passing through the lower pivot
            else:
                m1 = self._slopingUpperMax
                # Find intercept from equation of line passing through upper pivot: 
                #   b1 = y - m1*x
                b1 = self._lowerPivot.value - m1*self._lowerPivot.time 

            # L2 will be the line between the last two points
            # Find gradient betwen this point and last point
            m2 = (thisPoint.value - self._lastPoints[0].value)/(thisPoint.time - self._lastPoints[0].time)
            # Find intercept from equation of line passing through this point:
            #   b2 = y - m2*x
            b2 = thisPoint.value - m2 * thisPoint.time                    

            # Find point of intersection between L1 and L2 
            # which will be the upper boundary for the parallelogram 
            newPoint = FilterPoint((b2 - b1)/(m1 - m2), m1*(b2 - b1)/(m1 - m2) + b1)
            # Offset point to lie between the upper and lower boundaries of the parallelogram
            if(slopingUpperMaxUpdated):
                newPoint += FilterPoint(
                                0, 
                                -self._compressionDeviation/2 if slopingUpperMaxUpdated \
                                    else self._compressionDeviation/2
                                )

            # New point generated by compression algorithm
            results += [(newPoint.time, newPoint.value)]
            
            # Recalculate the window
            self._updateWindow(thisPoint, newPoint)
            self._firstTime = newPoint.time

            # Re-evaluate the current point
            results += self._evaluateParallelogram(thisPoint)            

        return results

{% endhighlight %}

The *_updateWindow()* method shown below resets the state of the SDT algorithm for a new envelope. It calculates new upper and lower pivot values from the output point generated previously in the call to *_evaluateParallelogram()*. The *_slopingUpperMax* and *_slopingLowerMin* values are then recalculated as the gradients between the last input point and the upper and lower pivot points respectively.

{% highlight python %}

    def _updateWindow(self, thisPoint, newPoint):
        """ Initialises window for a new parallelogram. """
        
        # Update upper and lower pivot
        self._upperPivot = newPoint + FilterPoint(0, self._compressionDeviation)
        self._lowerPivot = newPoint + FilterPoint(0, -self._compressionDeviation)
        
        # Update the sloping upper and sloping lower gradients
        # Sloping upper is the gradient between the current point and the upper pivot
        slopingUpper = (thisPoint.value - self._upperPivot.value)/(thisPoint.time - self._upperPivot.time)
        # Sloping lower is the gradient between the current point and the lower pivot 
        slopingLower = (thisPoint.value - self._lowerPivot.value)/(thisPoint.time - self._lowerPivot.time)
        
        # Uppdate limits on sloping upper and sloping lower gradients
        self._slopingUpperMax = slopingUpper
        self._slopingLowerMin = slopingLower

        return
{% endhighlight %}

Finally, the *flush()* method below may be called to return the last input point presented to the algorithm which may be needed if the last input point is required to be included in the output data set. This could be the case if processing a batch of points.

{% highlight python %}

    def flush(self) -> list:
        """ Returns the last point if available. """
        results = []
        
        # The first point is always returned, so need at least two last points
        if(len(self._lastPoints) > 1):
            results += [(self._lastPoints[0].time, self._lastPoints[0].value)]
            self._updateWindow(self._lastPoints[-1], self._lastPoints[0]) 
        
        return results  
{% endhighlight %}

An example is provided below of applying compression on a point-by-point basis using *filterPoint()* method of the *SdtFilter* class.

{% highlight python %}
import matplotlib.pyplot as plt
from pydbfilter import SdtFilter

# Initialize some data
input_time = [0, 5, 10, 15, 20]
input_data = [0, 0.1, 1.6, 1.63, 1.66] 

# Create a filter object
filter = SdtFilter(0.05, 100)

# Pass in the first point
output_point = filter.filterPoint(input_time[0], input_data[0])
print("First point is {0}".format(output_point))
output_points = output_point

# Pass in the second point
output_point = filter.filterPoint(input_time[1], input_data[1])
print("Second point is {0}".format(output_point))
output_points += output_point

# Pass in the third point
output_point = filter.filterPoint(input_time[2], input_data[2])
print("Third point is {0}".format(output_point))
output_points += output_point

# Pass in the fourth point
output_point = filter.filterPoint(input_time[3], input_data[3])
print("Fourth point is {0}".format(output_point))
output_points += output_point

# Pass in the fifth point
output_point = filter.filterPoint(input_time[4], input_data[4])
print("Fifth point is {0}".format(output_point))
output_points += output_point
{% endhighlight %}

The last input point may be flushed to the output using the *flush()* method:

{% highlight python %}
# Flush the last point
output_point = filter.flush()
print("Last point is {0}".format(output_point))
output_points += output_point
{% endhighlight %}

Plotting the output:

{% highlight python %}
# Plot the output
output_time = [t for (t, d) in output_points]
output_data = [d for (t, d) in output_points]
plt.plot(input_time, input_data, "o-", label="Input Data")
plt.plot(output_time, output_data, "d--", label="Output Data")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("SDT Single Points (Compression deviation = 0.05)")
plt.legend()
plt.show()
{% endhighlight %}

The output data shows a reduced number of points with the distance of all input points to the interpolated output data within +/- the compression deviation.

![Trend showing output data from SDT in orange with input data in blue.](/assets/posts/swinging-door-trending-part-1/sdt8.png?raw=true)
{:.image-caption}
*Figure 8: Output of SDT algorithm after processing point sequentially.*

Points may be filtered as a batch using either a list or a Pandas DataFrame with the *filterPoints()* method.

For example, using a list:

{% highlight python %}
import matplotlib.pyplot as plt
from math import sin, pi
from pydbfilter import SdtFilter, DeadbandFilter, FilterTree

# Generate a sine wave
input_time = [i for i in range(0,40)]
input_data = [sin(t*2*pi/20) for t in input_time] 
input_points = list(zip(input_time, input_data))

# Create a filter object
filter = SdtFilter(0.05, 100)

# Filter the points
output_points = filter.filterPoints(input_points)

# Flush the last point from the filter
output_points += filter.flush()

# Plot the output
output_time = [t for (t, d) in output_points]
output_data = [d for (t, d) in output_points]
plt.plot(input_time, input_data, "o-", label="Input Data")
plt.plot(output_time, output_data, "d--", label="Output Data")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("SDT Batch of Points (Compression deviation = 0.05)")
plt.legend()
plt.show()
{% endhighlight %}

Results in the following output:

![Trend showing compressed output data from SDT in orange with input sine function in blue.](/assets/posts/swinging-door-trending-part-1/sdt9.png?raw=true)
{:.image-caption}
*Figure 9: Output of SDT algorithm after processing points in a batch.*

A DataFrame object may also be passed to *filterPoints()*. In this case the first column will be used as time values and the second column as the magnitude.

An example using a DataFrame object is shown below.

{% highlight python %}
import matplotlib.pyplot as plt
from math import sin, pi
from pandas import DataFrame
from pydbfilter import SdtFilter, DeadbandFilter, FilterTree

# Generate a sine wave
input_time = [i for i in range(0,40)]
input_data = [sin(t*2*pi/20) for t in input_time] 
input_points = DataFrame({
                't': input_time,
                'v' : input_data
                })

# Create a filter object
filter = SdtFilter(0.05, 100)

# Filter the points
output_points = filter.filterPoints(input_points)

# Flush the last point from the filter
flushed_points = filter.flush()
output_points = output_points.append({
    't' : flushed_points[0][0],
    'v' : flushed_points[0][1]
    }, ignore_index=True)

# Plot the output
output_time = output_points['t']
output_data = output_points['v']
plt.plot(input_time, input_data, "o-", label="Input Data")
plt.plot(output_time, output_data, "d--", label="Output Data")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.title("SDT Batch of Points (Compression deviation = 0.05)")
plt.legend()
plt.show()
{% endhighlight %}


## Python Package and GitHub Repository

The Python package and example scripts are available on my GitHub respository [located here](https://github.com/bott-j/pydbfilter). 

## References

<a id="1">[1]</a> 
J. D. A. Correa, C. Montez, A. S. R. Pinto and E. M. Leao, “Swinging Door Trending Compression Algorithm for IoT Environments,” IX Simpósio Brasileiro de Engenharia de Sistemas Computacionais, 2019.


