	؁sF??S@؁sF??S@!؁sF??S@	?f?r????f?r???!?f?r???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$؁sF??S@??0?*??A?Y??ڎS@Y??H.?!??*	    ?I@2F
Iterator::ModelD?l?????!jiiii?P@)?e??a???1??????E@:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCacheL7?A`???!------@@)vq?-??1??????>@:Preprocessing2P
Iterator::Model::Prefetcha??+e??!PPPPPP8@)a??+e??1PPPPPP8@:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImplǺ???F?!????????)Ǻ???F?1????????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?f?r???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??0?*????0?*??!??0?*??      ??!       "      ??!       *      ??!       2	?Y??ڎS@?Y??ڎS@!?Y??ڎS@:      ??!       B      ??!       J	??H.?!????H.?!??!??H.?!??R      ??!       Z	??H.?!????H.?!??!??H.?!??JCPU_ONLYY?f?r???b 