	??	h"DZ@??	h"DZ@!??	h"DZ@	pN???-??pN???-??!pN???-??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??	h"DZ@?C??????A^?I3Z@Y?St$????*	gfffffG@2F
Iterator::Model6?;Nё??!??N???M@)a2U0*???14H?4H?D@:Preprocessing2]
&Iterator::Model::Prefetch::MemoryCache?ݓ??Z??!;?;1D@)e?X???1?????{B@:Preprocessing2P
Iterator::Model::Prefetch?5?;Nс?!p??o??2@)?5?;Nс?1p??o??2@:Preprocessing2a
*Iterator::Model::Prefetch::MemoryCacheImpl-C??6Z?!??Y??Y@)-C??6Z?1??Y??Y@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9oN???-??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?C???????C??????!?C??????      ??!       "      ??!       *      ??!       2	^?I3Z@^?I3Z@!^?I3Z@:      ??!       B      ??!       J	?St$?????St$????!?St$????R      ??!       Z	?St$?????St$????!?St$????JCPU_ONLYYoN???-??b 