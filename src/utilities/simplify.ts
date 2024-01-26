// @ts-nocheck

export let simplify: (
    data: {
        positions: [number, number, number][];
        colors?: [number, number, number][];
        cells: [number, number, number][];
    },
    threshold?: float,
) => (targetCount: int) => {
    positions: [number, number, number][];
    colors?: [number, number, number][];
    cells: [number, number, number][];
};

(function () {
    /////////////////////HELPERS/////////////////////
    var Heap;
    (function () {
        var defaultCmp,
            floor,
            heapify,
            heappop,
            heappush,
            heappushpop,
            heapreplace,
            insort,
            min,
            nlargest,
            nsmallest,
            updateItem,
            _siftdown,
            _siftup;

        (floor = Math.floor), (min = Math.min);

        defaultCmp = function (x, y) {
            if (x < y) {
                return -1;
            }
            if (x > y) {
                return 1;
            }
            return 0;
        };

        insort = function (a, x, lo, hi, cmp) {
            var mid;
            if (lo == null) {
                lo = 0;
            }
            if (cmp == null) {
                cmp = defaultCmp;
            }
            if (lo < 0) {
                throw new Error("lo must be non-negative");
            }
            if (hi == null) {
                hi = a.length;
            }
            while (lo < hi) {
                mid = floor((lo + hi) / 2);
                if (cmp(x, a[mid]) < 0) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            return [].splice.apply(a, [lo, lo - lo].concat(x)), x;
        };

        heappush = function (array, item, cmp) {
            if (cmp == null) {
                cmp = defaultCmp;
            }
            array.push(item);
            return _siftdown(array, 0, array.length - 1, cmp);
        };

        heappop = function (array, cmp) {
            var lastelt, returnitem;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            lastelt = array.pop();
            if (array.length) {
                returnitem = array[0];
                array[0] = lastelt;
                _siftup(array, 0, cmp);
            } else {
                returnitem = lastelt;
            }
            return returnitem;
        };

        heapreplace = function (array, item, cmp) {
            var returnitem;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            returnitem = array[0];
            array[0] = item;
            _siftup(array, 0, cmp);
            return returnitem;
        };

        heappushpop = function (array, item, cmp) {
            var _ref;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            if (array.length && cmp(array[0], item) < 0) {
                (_ref = [array[0], item]),
                    (item = _ref[0]),
                    (array[0] = _ref[1]);
                _siftup(array, 0, cmp);
            }
            return item;
        };

        heapify = function (array, cmp) {
            var i, _i, _j, _len, _ref, _ref1, _results, _results1;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            _ref1 = function () {
                _results1 = [];
                for (
                    var _j = 0, _ref = floor(array.length / 2);
                    0 <= _ref ? _j < _ref : _j > _ref;
                    0 <= _ref ? _j++ : _j--
                ) {
                    _results1.push(_j);
                }
                return _results1;
            }
                .apply(this)
                .reverse();
            _results = [];
            for (_i = 0, _len = _ref1.length; _i < _len; _i++) {
                i = _ref1[_i];
                _results.push(_siftup(array, i, cmp));
            }
            return _results;
        };

        updateItem = function (array, item, cmp) {
            var pos;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            pos = array.indexOf(item);
            if (pos === -1) {
                return;
            }
            _siftdown(array, 0, pos, cmp);
            return _siftup(array, pos, cmp);
        };

        nlargest = function (array, n, cmp) {
            var elem, result, _i, _len, _ref;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            result = array.slice(0, n);
            if (!result.length) {
                return result;
            }
            heapify(result, cmp);
            _ref = array.slice(n);
            for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                elem = _ref[_i];
                heappushpop(result, elem, cmp);
            }
            return result.sort(cmp).reverse();
        };

        nsmallest = function (array, n, cmp) {
            var elem, i, los, result, _i, _j, _len, _ref, _ref1, _results;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            if (n * 10 <= array.length) {
                result = array.slice(0, n).sort(cmp);
                if (!result.length) {
                    return result;
                }
                los = result[result.length - 1];
                _ref = array.slice(n);
                for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                    elem = _ref[_i];
                    if (cmp(elem, los) < 0) {
                        insort(result, elem, 0, null, cmp);
                        result.pop();
                        los = result[result.length - 1];
                    }
                }
                return result;
            }
            heapify(array, cmp);
            _results = [];
            for (
                i = _j = 0, _ref1 = min(n, array.length);
                0 <= _ref1 ? _j < _ref1 : _j > _ref1;
                i = 0 <= _ref1 ? ++_j : --_j
            ) {
                _results.push(heappop(array, cmp));
            }
            return _results;
        };

        _siftdown = function (array, startpos, pos, cmp) {
            var newitem, parent, parentpos;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            newitem = array[pos];
            while (pos > startpos) {
                parentpos = (pos - 1) >> 1;
                parent = array[parentpos];
                if (cmp(newitem, parent) < 0) {
                    array[pos] = parent;
                    pos = parentpos;
                    continue;
                }
                break;
            }
            return (array[pos] = newitem);
        };

        _siftup = function (array, pos, cmp) {
            var childpos, endpos, newitem, rightpos, startpos;
            if (cmp == null) {
                cmp = defaultCmp;
            }
            endpos = array.length;
            startpos = pos;
            newitem = array[pos];
            childpos = 2 * pos + 1;
            while (childpos < endpos) {
                rightpos = childpos + 1;
                if (
                    rightpos < endpos &&
                    !(cmp(array[childpos], array[rightpos]) < 0)
                ) {
                    childpos = rightpos;
                }
                array[pos] = array[childpos];
                pos = childpos;
                childpos = 2 * pos + 1;
            }
            array[pos] = newitem;
            return _siftdown(array, startpos, pos, cmp);
        };

        Heap = (function () {
            Heap.push = heappush;
            Heap.pop = heappop;
            Heap.replace = heapreplace;
            Heap.pushpop = heappushpop;
            Heap.heapify = heapify;
            Heap.updateItem = updateItem;
            Heap.nlargest = nlargest;
            Heap.nsmallest = nsmallest;

            function Heap(cmp) {
                this.cmp = cmp != null ? cmp : defaultCmp;
                this.nodes = [];
            }

            Heap.prototype.push = function (x) {
                return heappush(this.nodes, x, this.cmp);
            };
            Heap.prototype.pop = function () {
                return heappop(this.nodes, this.cmp);
            };
            Heap.prototype.peek = function () {
                return this.nodes[0];
            };
            Heap.prototype.contains = function (x) {
                return this.nodes.indexOf(x) !== -1;
            };
            Heap.prototype.replace = function (x) {
                return heapreplace(this.nodes, x, this.cmp);
            };
            Heap.prototype.pushpop = function (x) {
                return heappushpop(this.nodes, x, this.cmp);
            };
            Heap.prototype.heapify = function () {
                return heapify(this.nodes, this.cmp);
            };
            Heap.prototype.updateItem = function (x) {
                return updateItem(this.nodes, x, this.cmp);
            };
            Heap.prototype.clear = function () {
                return (this.nodes = []);
            };
            Heap.prototype.empty = function () {
                return this.nodes.length === 0;
            };
            Heap.prototype.size = function () {
                return this.nodes.length;
            };
            Heap.prototype.toArray = function () {
                return this.nodes.slice(0);
            };

            Heap.prototype.insert = Heap.prototype.push;
            Heap.prototype.top = Heap.prototype.peek;
            Heap.prototype.front = Heap.prototype.peek;
            Heap.prototype.has = Heap.prototype.contains;
            Heap.prototype.copy = Heap.prototype.clone;

            return Heap;
        })();
    })();

    var vec4;
    (function () {
        function vec4transformMat4(out, a, m) {
            var x = a[0],
                y = a[1],
                z = a[2],
                w = a[3];
            out[0] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
            out[1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
            out[2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
            out[3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
            return out;
        }

        function vec4dot(a, b) {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        }

        vec4 = {
            transformMat4: vec4transformMat4,
            dot: vec4dot,
        };
    })();

    var vec3;
    (function () {
        function vec3add(out, a, b) {
            out[0] = a[0] + b[0];
            out[1] = a[1] + b[1];
            out[2] = a[2] + b[2];
            return out;
        }

        function vec3scale(out, a, b) {
            out[0] = a[0] * b;
            out[1] = a[1] * b;
            out[2] = a[2] * b;
            return out;
        }

        function vec3distance(a, b) {
            var x = b[0] - a[0],
                y = b[1] - a[1],
                z = b[2] - a[2];
            return Math.sqrt(x * x + y * y + z * z);
        }

        function vec3dot(a, b) {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        vec3 = {
            add: vec3add,
            scale: vec3scale,
            distance: vec3distance,
            dot: vec3dot,
        };
    })();

    var normals;
    (function () {
        normals = function (faces, positions, specifiedEpsilon?) {
            var DEFAULT_FACE_EPSILON = 1e-6;
            var N = faces.length;
            var norms = new Array(N);
            var epsilon =
                specifiedEpsilon === void 0
                    ? DEFAULT_FACE_EPSILON
                    : specifiedEpsilon;

            for (var i = 0; i < N; ++i) {
                var f = faces[i];
                var pos = new Array(3);
                for (var j = 0; j < 3; ++j) {
                    pos[j] = positions[f[j]];
                }

                var d01 = new Array(3);
                var d21 = new Array(3);
                for (var j = 0; j < 3; ++j) {
                    d01[j] = pos[1][j] - pos[0][j];
                    d21[j] = pos[2][j] - pos[0][j];
                }

                var n = new Array(3);
                var l = 0.0;
                for (var j = 0; j < 3; ++j) {
                    var u = (j + 1) % 3;
                    var v = (j + 2) % 3;
                    n[j] = d01[u] * d21[v] - d01[v] * d21[u];
                    l += n[j] * n[j];
                }
                if (l > epsilon) {
                    l = 1.0 / Math.sqrt(l);
                } else {
                    l = 0.0;
                }
                for (var j = 0; j < 3; ++j) {
                    n[j] *= l;
                }
                norms[i] = n;
            }
            return norms;
        };
    })();

    var removeOrphans;
    (function () {
        removeOrphans = function (cells, positions, colors) {
            var newPositions = [];
            var newColors = [];
            var indexLookup = {};

            var newCells = cells.map(function (cell) {
                return cell.map(function (index) {
                    if (indexLookup[index] === undefined) {
                        indexLookup[index] = newPositions.length;
                        newPositions.push(positions[index]);
                        newColors.push(colors[index]);
                    }
                    return indexLookup[index];
                });
            });

            return {
                cells: newCells,
                positions: newPositions,
                colors: newColors,
            };
        };
    })();

    var removeDegenerates;
    (function () {
        function arrayEqual(a, b) {
            if (a.length !== b.length) {
                return false;
            }
            for (var i = 0; i < a.length; i++) {
                if (a[i] !== b[i]) {
                    return false;
                }
            }
            return true;
        }
        function integerEqual(a, b) {
            return a === b;
        }
        removeDegenerates = function (cells, positions?) {
            var equal = integerEqual;
            if (positions) {
                equal = arrayEqual;
            }
            return cells.filter(function (cell) {
                if (positions) {
                    cell = cell.map(function (index) {
                        return positions[index];
                    });
                }
                for (var i = 0; i < cell.length; i++) {
                    for (var j = 0; j < cell.length; j++) {
                        if (i != j && equal(cell[i], cell[j])) {
                            return false;
                        }
                    }
                }
                return true;
            });
        };
    })();

    var iota;
    (function () {
        iota = function (n) {
            var result = new Array(n);
            for (var i = 0; i < n; ++i) {
                result[i] = i;
            }
            return result;
        };
    })();

    var isBuffer;
    (function () {
        function isBufferA(obj) {
            return (
                !!obj.constructor &&
                typeof obj.constructor.isBuffer === "function" &&
                obj.constructor.isBuffer(obj)
            );
        }
        // For Node v0.10 support. Remove this eventually.
        function isSlowBuffer(obj) {
            return (
                typeof obj.readFloatLE === "function" &&
                typeof obj.slice === "function" &&
                isBufferA(obj.slice(0, 0))
            );
        }
        isBuffer = function (obj) {
            return (
                obj != null &&
                (isBufferA(obj) || isSlowBuffer(obj) || !!obj._isBuffer)
            );
        };
    })();

    var ndarray;
    (function () {
        var hasTypedArrays = typeof Float64Array !== "undefined";

        function compare1st(a, b) {
            return a[0] - b[0];
        }

        function order() {
            var stride = this.stride;
            var terms = new Array(stride.length);
            var i;
            for (i = 0; i < terms.length; ++i) {
                terms[i] = [Math.abs(stride[i]), i];
            }
            terms.sort(compare1st);
            var result = new Array(terms.length);
            for (i = 0; i < result.length; ++i) {
                result[i] = terms[i][1];
            }
            return result;
        }

        function compileConstructor(dtype, dimension) {
            var className = ["View", dimension, "d", dtype].join("");
            if (dimension < 0) {
                className = "View_Nil" + dtype;
            }
            var useGetters = dtype === "generic";
            var code;

            if (dimension === -1) {
                //Special case for trivial arrays
                code =
                    "function " +
                    className +
                    "(a){this.data=a;};\
                var proto=" +
                    className +
                    ".prototype;\
                proto.dtype='" +
                    dtype +
                    "';\
                proto.index=function(){return -1};\
                proto.size=0;\
                proto.dimension=-1;\
                proto.shape=proto.stride=proto.order=[];\
                proto.lo=proto.hi=proto.transpose=proto.step=\
                function(){return new " +
                    className +
                    "(this.data);};\
                proto.get=proto.set=function(){};\
                proto.pick=function(){return null};\
                return function construct_" +
                    className +
                    "(a){return new " +
                    className +
                    "(a);}";
                var procedure = new Function(code);
                return procedure();
            } else if (dimension === 0) {
                //Special case for 0d arrays
                code =
                    "function " +
                    className +
                    "(a,d) {\
                this.data = a;\
                this.offset = d\
                };\
                var proto=" +
                    className +
                    ".prototype;\
                proto.dtype='" +
                    dtype +
                    "';\
                proto.index=function(){return this.offset};\
                proto.dimension=0;\
                proto.size=1;\
                proto.shape=\
                proto.stride=\
                proto.order=[];\
                proto.lo=\
                proto.hi=\
                proto.transpose=\
                proto.step=function " +
                    className +
                    "_copy() {\
                return new " +
                    className +
                    "(this.data,this.offset)\
                };\
                proto.pick=function " +
                    className +
                    "_pick(){\
                return TrivialArray(this.data);\
                };\
                proto.valueOf=proto.get=function " +
                    className +
                    "_get(){\
                return " +
                    (useGetters
                        ? "this.data.get(this.offset)"
                        : "this.data[this.offset]") +
                    "};\
                proto.set=function " +
                    className +
                    "_set(v){\
                return " +
                    (useGetters
                        ? "this.data.set(this.offset,v)"
                        : "this.data[this.offset]=v") +
                    "\
                };\
                return function construct_" +
                    className +
                    "(a,b,c,d){return new " +
                    className +
                    "(a,d)}";
                var procedure = new Function("TrivialArray", code);
                return procedure(CACHED_CONSTRUCTORS[dtype][0]);
            }

            code = ["'use strict'"];

            //Create constructor for view
            var indices = iota(dimension);
            var args = indices.map(function (i) {
                return "i" + i;
            });
            var index_str =
                "this.offset+" +
                indices
                    .map(function (i) {
                        return "this.stride[" + i + "]*i" + i;
                    })
                    .join("+");
            var shapeArg = indices
                .map(function (i) {
                    return "b" + i;
                })
                .join(",");
            var strideArg = indices
                .map(function (i) {
                    return "c" + i;
                })
                .join(",");
            code.push(
                "function " +
                    className +
                    "(a," +
                    shapeArg +
                    "," +
                    strideArg +
                    ",d){this.data=a",
                "this.shape=[" + shapeArg + "]",
                "this.stride=[" + strideArg + "]",
                "this.offset=d|0}",
                "var proto=" + className + ".prototype",
                "proto.dtype='" + dtype + "'",
                "proto.dimension=" + dimension,
            );

            //view.size:
            code.push(
                "Object.defineProperty(proto,'size',{get:function " +
                    className +
                    "_size(){\
return " +
                    indices
                        .map(function (i) {
                            return "this.shape[" + i + "]";
                        })
                        .join("*"),
                "}})",
            );

            //view.order:
            if (dimension === 1) {
                code.push("proto.order=[0]");
            } else {
                code.push("Object.defineProperty(proto,'order',{get:");
                if (dimension < 4) {
                    code.push("function " + className + "_order(){");
                    if (dimension === 2) {
                        code.push(
                            "return (Math.abs(this.stride[0])>Math.abs(this.stride[1]))?[1,0]:[0,1]}})",
                        );
                    } else if (dimension === 3) {
                        code.push(
                            "var s0=Math.abs(this.stride[0]),s1=Math.abs(this.stride[1]),s2=Math.abs(this.stride[2]);\
                        if(s0>s1){\
                        if(s1>s2){\
                        return [2,1,0];\
                        }else if(s0>s2){\
                        return [1,2,0];\
                        }else{\
                        return [1,0,2];\
                        }\
                        }else if(s0>s2){\
                        return [2,0,1];\
                        }else if(s2>s1){\
                        return [0,1,2];\
                        }else{\
                        return [0,2,1];\
                        }}})",
                        );
                    }
                } else {
                    code.push("ORDER})");
                }
            }

            //view.set(i0, ..., v):
            code.push(
                "proto.set=function " +
                    className +
                    "_set(" +
                    args.join(",") +
                    ",v){",
            );
            if (useGetters) {
                code.push("return this.data.set(" + index_str + ",v)}");
            } else {
                code.push("return this.data[" + index_str + "]=v}");
            }

            //view.get(i0, ...):
            code.push(
                "proto.get=function " +
                    className +
                    "_get(" +
                    args.join(",") +
                    "){",
            );
            if (useGetters) {
                code.push("return this.data.get(" + index_str + ")}");
            } else {
                code.push("return this.data[" + index_str + "]}");
            }

            //view.index:
            code.push(
                "proto.index=function " + className + "_index(",
                args.join(),
                "){return " + index_str + "}",
            );

            //view.hi():
            code.push(
                "proto.hi=function " +
                    className +
                    "_hi(" +
                    args.join(",") +
                    "){return new " +
                    className +
                    "(this.data," +
                    indices
                        .map(function (i) {
                            return [
                                "(typeof i",
                                i,
                                "!=='number'||i",
                                i,
                                "<0)?this.shape[",
                                i,
                                "]:i",
                                i,
                                "|0",
                            ].join("");
                        })
                        .join(",") +
                    "," +
                    indices
                        .map(function (i) {
                            return "this.stride[" + i + "]";
                        })
                        .join(",") +
                    ",this.offset)}",
            );

            //view.lo():
            var a_vars = indices.map(function (i) {
                return "a" + i + "=this.shape[" + i + "]";
            });
            var c_vars = indices.map(function (i) {
                return "c" + i + "=this.stride[" + i + "]";
            });
            code.push(
                "proto.lo=function " +
                    className +
                    "_lo(" +
                    args.join(",") +
                    "){var b=this.offset,d=0," +
                    a_vars.join(",") +
                    "," +
                    c_vars.join(","),
            );
            for (var i = 0; i < dimension; ++i) {
                code.push(
                    "if(typeof i" +
                        i +
                        "==='number'&&i" +
                        i +
                        ">=0){\
                    d=i" +
                        i +
                        "|0;\
                    b+=c" +
                        i +
                        "*d;\
                    a" +
                        i +
                        "-=d}",
                );
            }
            code.push(
                "return new " +
                    className +
                    "(this.data," +
                    indices
                        .map(function (i) {
                            return "a" + i;
                        })
                        .join(",") +
                    "," +
                    indices
                        .map(function (i) {
                            return "c" + i;
                        })
                        .join(",") +
                    ",b)}",
            );

            //view.step():
            code.push(
                "proto.step=function " +
                    className +
                    "_step(" +
                    args.join(",") +
                    "){var " +
                    indices
                        .map(function (i) {
                            return "a" + i + "=this.shape[" + i + "]";
                        })
                        .join(",") +
                    "," +
                    indices
                        .map(function (i) {
                            return "b" + i + "=this.stride[" + i + "]";
                        })
                        .join(",") +
                    ",c=this.offset,d=0,ceil=Math.ceil",
            );
            for (var i = 0; i < dimension; ++i) {
                code.push(
                    "if(typeof i" +
                        i +
                        "==='number'){\
                    d=i" +
                        i +
                        "|0;\
                    if(d<0){\
                    c+=b" +
                        i +
                        "*(a" +
                        i +
                        "-1);\
                    a" +
                        i +
                        "=ceil(-a" +
                        i +
                        "/d)\
                    }else{\
                    a" +
                        i +
                        "=ceil(a" +
                        i +
                        "/d)\
                    }\
                    b" +
                        i +
                        "*=d\
                }",
                );
            }
            code.push(
                "return new " +
                    className +
                    "(this.data," +
                    indices
                        .map(function (i) {
                            return "a" + i;
                        })
                        .join(",") +
                    "," +
                    indices
                        .map(function (i) {
                            return "b" + i;
                        })
                        .join(",") +
                    ",c)}",
            );

            //view.transpose():
            var tShape = new Array(dimension);
            var tStride = new Array(dimension);
            for (var i = 0; i < dimension; ++i) {
                tShape[i] = "a[i" + i + "]";
                tStride[i] = "b[i" + i + "]";
            }
            code.push(
                "proto.transpose=function " +
                    className +
                    "_transpose(" +
                    args +
                    "){" +
                    args
                        .map(function (n, idx) {
                            return (
                                n +
                                "=(" +
                                n +
                                "===undefined?" +
                                idx +
                                ":" +
                                n +
                                "|0)"
                            );
                        })
                        .join(";"),
                "var a=this.shape,b=this.stride;return new " +
                    className +
                    "(this.data," +
                    tShape.join(",") +
                    "," +
                    tStride.join(",") +
                    ",this.offset)}",
            );

            //view.pick():
            code.push(
                "proto.pick=function " +
                    className +
                    "_pick(" +
                    args +
                    "){var a=[],b=[],c=this.offset",
            );
            for (var i = 0; i < dimension; ++i) {
                code.push(
                    "if(typeof i" +
                        i +
                        "==='number'&&i" +
                        i +
                        ">=0){c=(c+this.stride[" +
                        i +
                        "]*i" +
                        i +
                        ")|0}else{a.push(this.shape[" +
                        i +
                        "]);b.push(this.stride[" +
                        i +
                        "])}",
                );
            }
            code.push(
                "var ctor=CTOR_LIST[a.length+1];return ctor(this.data,a,b,c)}",
            );

            //Add return statement
            code.push(
                "return function construct_" +
                    className +
                    "(data,shape,stride,offset){return new " +
                    className +
                    "(data," +
                    indices
                        .map(function (i) {
                            return "shape[" + i + "]";
                        })
                        .join(",") +
                    "," +
                    indices
                        .map(function (i) {
                            return "stride[" + i + "]";
                        })
                        .join(",") +
                    ",offset)}",
            );

            //Compile procedure
            var procedure = new Function("CTOR_LIST", "ORDER", code.join("\n"));
            return procedure(CACHED_CONSTRUCTORS[dtype], order);
        }

        function arrayDType(data) {
            if (isBuffer(data)) {
                return "buffer";
            }
            if (hasTypedArrays) {
                switch (Object.prototype.toString.call(data)) {
                    case "[object Float64Array]":
                        return "float64";
                    case "[object Float32Array]":
                        return "float32";
                    case "[object Int8Array]":
                        return "int8";
                    case "[object Int16Array]":
                        return "int16";
                    case "[object Int32Array]":
                        return "int32";
                    case "[object Uint8Array]":
                        return "uint8";
                    case "[object Uint16Array]":
                        return "uint16";
                    case "[object Uint32Array]":
                        return "uint32";
                    case "[object Uint8ClampedArray]":
                        return "uint8_clamped";
                    case "[object BigInt64Array]":
                        return "bigint64";
                    case "[object BigUint64Array]":
                        return "biguint64";
                }
            }
            if (Array.isArray(data)) {
                return "array";
            }
            return "generic";
        }

        var CACHED_CONSTRUCTORS = {
            float32: [],
            float64: [],
            int8: [],
            int16: [],
            int32: [],
            uint8: [],
            uint16: [],
            uint32: [],
            array: [],
            uint8_clamped: [],
            bigint64: [],
            biguint64: [],
            buffer: [],
            generic: [],
        };

        (function () {
            for (var id in CACHED_CONSTRUCTORS) {
                CACHED_CONSTRUCTORS[id].push(compileConstructor(id, -1));
            }
        });

        function wrappedNDArrayCtor(data, shape, stride, offset) {
            if (data === undefined) {
                var ctor = CACHED_CONSTRUCTORS.array[0];
                return ctor([]);
            } else if (typeof data === "number") {
                data = [data];
            }
            if (shape === undefined) {
                shape = [data.length];
            }
            var d = shape.length;
            if (stride === undefined) {
                stride = new Array(d);
                for (var i = d - 1, sz = 1; i >= 0; --i) {
                    stride[i] = sz;
                    sz *= shape[i];
                }
            }
            if (offset === undefined) {
                offset = 0;
                for (var i = 0; i < d; ++i) {
                    if (stride[i] < 0) {
                        offset -= (shape[i] - 1) * stride[i];
                    }
                }
            }
            var dtype = arrayDType(data);
            var ctor_list = CACHED_CONSTRUCTORS[dtype];
            while (ctor_list.length <= d + 1) {
                ctor_list.push(compileConstructor(dtype, ctor_list.length - 1));
            }
            var ctor = ctor_list[d + 1];
            return ctor(data, shape, stride, offset);
        }

        ndarray = wrappedNDArrayCtor;
    })();

    var ops = {} as any;
    (function () {
        var createThunk;
        (function () {
            var uniq;
            (function () {
                function unique_pred(list, compare) {
                    var ptr = 1,
                        len = list.length,
                        a = list[0],
                        b = list[0];
                    for (var i = 1; i < len; ++i) {
                        b = a;
                        a = list[i];
                        if (compare(a, b)) {
                            if (i === ptr) {
                                ptr++;
                                continue;
                            }
                            list[ptr++] = a;
                        }
                    }
                    list.length = ptr;
                    return list;
                }

                function unique_eq(list) {
                    var ptr = 1,
                        len = list.length,
                        a = list[0],
                        b = list[0];
                    for (var i = 1; i < len; ++i, b = a) {
                        b = a;
                        a = list[i];
                        if (a !== b) {
                            if (i === ptr) {
                                ptr++;
                                continue;
                            }
                            list[ptr++] = a;
                        }
                    }
                    list.length = ptr;
                    return list;
                }

                uniq = function (list, compare, sorted) {
                    if (list.length === 0) {
                        return list;
                    }
                    if (compare) {
                        if (!sorted) {
                            list.sort(compare);
                        }
                        return unique_pred(list, compare);
                    }
                    if (!sorted) {
                        list.sort();
                    }
                    return unique_eq(list);
                };
            })();

            var compile;
            (function () {
                // This function generates very simple loops analogous to how you typically traverse arrays (the outermost loop corresponds to the slowest changing index, the innermost loop to the fastest changing index)
                // TODO: If two arrays have the same strides (and offsets) there is potential for decreasing the number of "pointers" and related variables. The drawback is that the type signature would become more specific and that there would thus be less potential for caching, but it might still be worth it, especially when dealing with large numbers of arguments.
                function innerFill(order, proc, body) {
                    var dimension = order.length,
                        nargs = proc.arrayArgs.length,
                        has_index = proc.indexArgs.length > 0,
                        code = [],
                        vars = [],
                        idx = 0,
                        pidx = 0,
                        i,
                        j;
                    for (i = 0; i < dimension; ++i) {
                        // Iteration variables
                        vars.push(["i", i, "=0"].join(""));
                    }
                    //Compute scan deltas
                    for (j = 0; j < nargs; ++j) {
                        for (i = 0; i < dimension; ++i) {
                            pidx = idx;
                            idx = order[i];
                            if (i === 0) {
                                // The innermost/fastest dimension's delta is simply its stride
                                vars.push(
                                    ["d", j, "s", i, "=t", j, "p", idx].join(
                                        "",
                                    ),
                                );
                            } else {
                                // For other dimensions the delta is basically the stride minus something which essentially "rewinds" the previous (more inner) dimension
                                vars.push(
                                    [
                                        "d",
                                        j,
                                        "s",
                                        i,
                                        "=(t",
                                        j,
                                        "p",
                                        idx,
                                        "-s",
                                        pidx,
                                        "*t",
                                        j,
                                        "p",
                                        pidx,
                                        ")",
                                    ].join(""),
                                );
                            }
                        }
                    }
                    if (vars.length > 0) {
                        code.push("var " + vars.join(","));
                    }
                    //Scan loop
                    for (i = dimension - 1; i >= 0; --i) {
                        // Start at largest stride and work your way inwards
                        idx = order[i];
                        code.push(
                            [
                                "for(i",
                                i,
                                "=0;i",
                                i,
                                "<s",
                                idx,
                                ";++i",
                                i,
                                "){",
                            ].join(""),
                        );
                    }
                    //Push body of inner loop
                    code.push(body);
                    //Advance scan pointers
                    for (i = 0; i < dimension; ++i) {
                        pidx = idx;
                        idx = order[i];
                        for (j = 0; j < nargs; ++j) {
                            code.push(["p", j, "+=d", j, "s", i].join(""));
                        }
                        if (has_index) {
                            if (i > 0) {
                                code.push(
                                    ["index[", pidx, "]-=s", pidx].join(""),
                                );
                            }
                            code.push(["++index[", idx, "]"].join(""));
                        }
                        code.push("}");
                    }
                    return code.join("\n");
                }

                // Generate "outer" loops that loop over blocks of data, applying "inner" loops to the blocks by manipulating the local variables in such a way that the inner loop only "sees" the current block.
                // TODO: If this is used, then the previous declaration (done by generateCwiseOp) of s* is essentially unnecessary.
                //       I believe the s* are not used elsewhere (in particular, I don't think they're used in the pre/post parts and "shape" is defined independently), so it would be possible to make defining the s* dependent on what loop method is being used.
                function outerFill(matched, order, proc, body) {
                    var dimension = order.length,
                        nargs = proc.arrayArgs.length,
                        blockSize = proc.blockSize,
                        has_index = proc.indexArgs.length > 0,
                        code = [];
                    var i;
                    for (i = 0; i < nargs; ++i) {
                        code.push(["var offset", i, "=p", i].join(""));
                    }
                    //Generate loops for unmatched dimensions
                    // The order in which these dimensions are traversed is fairly arbitrary (from small stride to large stride, for the first argument)
                    // TODO: It would be nice if the order in which these loops are placed would also be somehow "optimal" (at the very least we should check that it really doesn't hurt us if they're not).
                    for (i = matched; i < dimension; ++i) {
                        code.push(
                            [
                                "for(var j" + i + "=SS[",
                                order[i],
                                "]|0;j",
                                i,
                                ">0;){",
                            ].join(""),
                        ); // Iterate back to front
                        code.push(["if(j", i, "<", blockSize, "){"].join("")); // Either decrease j by blockSize (s = blockSize), or set it to zero (after setting s = j).
                        code.push(["s", order[i], "=j", i].join(""));
                        code.push(["j", i, "=0"].join(""));
                        code.push(
                            ["}else{s", order[i], "=", blockSize].join(""),
                        );
                        code.push(["j", i, "-=", blockSize, "}"].join(""));
                        if (has_index) {
                            code.push(["index[", order[i], "]=j", i].join(""));
                        }
                    }
                    for (i = 0; i < nargs; ++i) {
                        var indexStr = ["offset" + i];
                        for (var j = matched; j < dimension; ++j) {
                            indexStr.push(
                                ["j", j, "*t", i, "p", order[j]].join(""),
                            );
                        }
                        code.push(
                            ["p", i, "=(", indexStr.join("+"), ")"].join(""),
                        );
                    }
                    code.push(innerFill(order, proc, body));
                    for (var i = matched; i < dimension; ++i) {
                        code.push("}");
                    }
                    return code.join("\n");
                }

                //Count the number of compatible inner orders
                // This is the length of the longest common prefix of the arrays in orders.
                // Each array in orders lists the dimensions of the correspond ndarray in order of increasing stride.
                // This is thus the maximum number of dimensions that can be efficiently traversed by simple nested loops for all arrays.
                function countMatches(orders) {
                    var matched = 0,
                        dimension = orders[0].length;
                    while (matched < dimension) {
                        for (var j = 1; j < orders.length; ++j) {
                            if (orders[j][matched] !== orders[0][matched]) {
                                return matched;
                            }
                        }
                        ++matched;
                    }
                    return matched;
                }

                //Processes a block according to the given data types
                // Replaces variable names by different ones, either "local" ones (that are then ferried in and out of the given array) or ones matching the arguments that the function performing the ultimate loop will accept.
                function processBlock(block, proc, dtypes) {
                    var code = block.body;
                    var pre = [];
                    var post = [];
                    for (var i = 0; i < block.args.length; ++i) {
                        var carg = block.args[i];
                        if (carg.count <= 0) {
                            continue;
                        }
                        var re = new RegExp(carg.name, "g");
                        var ptrStr = "";
                        var arrNum = proc.arrayArgs.indexOf(i);
                        switch (proc.argTypes[i]) {
                            case "offset":
                                var offArgIndex =
                                    proc.offsetArgIndex.indexOf(i);
                                var offArg = proc.offsetArgs[offArgIndex];
                                arrNum = offArg.array;
                                ptrStr = "+q" + offArgIndex; // Adds offset to the "pointer" in the array
                            case "array":
                                ptrStr = "p" + arrNum + ptrStr;
                                var localStr = "l" + i;
                                var arrStr = "a" + arrNum;
                                if (proc.arrayBlockIndices[arrNum] === 0) {
                                    // Argument to body is just a single value from this array
                                    if (carg.count === 1) {
                                        // Argument/array used only once(?)
                                        if (dtypes[arrNum] === "generic") {
                                            if (carg.lvalue) {
                                                pre.push(
                                                    [
                                                        "var ",
                                                        localStr,
                                                        "=",
                                                        arrStr,
                                                        ".get(",
                                                        ptrStr,
                                                        ")",
                                                    ].join(""),
                                                ); // Is this necessary if the argument is ONLY used as an lvalue? (keep in mind that we can have a += something, so we would actually need to check carg.rvalue)
                                                code = code.replace(
                                                    re,
                                                    localStr,
                                                );
                                                post.push(
                                                    [
                                                        arrStr,
                                                        ".set(",
                                                        ptrStr,
                                                        ",",
                                                        localStr,
                                                        ")",
                                                    ].join(""),
                                                );
                                            } else {
                                                code = code.replace(
                                                    re,
                                                    [
                                                        arrStr,
                                                        ".get(",
                                                        ptrStr,
                                                        ")",
                                                    ].join(""),
                                                );
                                            }
                                        } else {
                                            code = code.replace(
                                                re,
                                                [arrStr, "[", ptrStr, "]"].join(
                                                    "",
                                                ),
                                            );
                                        }
                                    } else if (dtypes[arrNum] === "generic") {
                                        pre.push(
                                            [
                                                "var ",
                                                localStr,
                                                "=",
                                                arrStr,
                                                ".get(",
                                                ptrStr,
                                                ")",
                                            ].join(""),
                                        ); // TODO: Could we optimize by checking for carg.rvalue?
                                        code = code.replace(re, localStr);
                                        if (carg.lvalue) {
                                            post.push(
                                                [
                                                    arrStr,
                                                    ".set(",
                                                    ptrStr,
                                                    ",",
                                                    localStr,
                                                    ")",
                                                ].join(""),
                                            );
                                        }
                                    } else {
                                        pre.push(
                                            [
                                                "var ",
                                                localStr,
                                                "=",
                                                arrStr,
                                                "[",
                                                ptrStr,
                                                "]",
                                            ].join(""),
                                        ); // TODO: Could we optimize by checking for carg.rvalue?
                                        code = code.replace(re, localStr);
                                        if (carg.lvalue) {
                                            post.push(
                                                [
                                                    arrStr,
                                                    "[",
                                                    ptrStr,
                                                    "]=",
                                                    localStr,
                                                ].join(""),
                                            );
                                        }
                                    }
                                } else {
                                    // Argument to body is a "block"
                                    var reStrArr = [carg.name],
                                        ptrStrArr = [ptrStr];
                                    for (
                                        var j = 0;
                                        j <
                                        Math.abs(
                                            proc.arrayBlockIndices[arrNum],
                                        );
                                        j++
                                    ) {
                                        reStrArr.push("\\s*\\[([^\\]]+)\\]");
                                        ptrStrArr.push(
                                            "$" +
                                                (j + 1) +
                                                "*t" +
                                                arrNum +
                                                "b" +
                                                j,
                                        ); // Matched index times stride
                                    }
                                    re = new RegExp(reStrArr.join(""), "g");
                                    ptrStr = ptrStrArr.join("+");
                                    if (dtypes[arrNum] === "generic") {
                                        /*if(carg.lvalue) {
              pre.push(["var ", localStr, "=", arrStr, ".get(", ptrStr, ")"].join("")) // Is this necessary if the argument is ONLY used as an lvalue? (keep in mind that we can have a += something, so we would actually need to check carg.rvalue)
              code = code.replace(re, localStr)
              post.push([arrStr, ".set(", ptrStr, ",", localStr,")"].join(""))
            } else {
              code = code.replace(re, [arrStr, ".get(", ptrStr, ")"].join(""))
            }*/
                                        throw new Error(
                                            "cwise: Generic arrays not supported in combination with blocks!",
                                        );
                                    } else {
                                        // This does not produce any local variables, even if variables are used multiple times. It would be possible to do so, but it would complicate things quite a bit.
                                        code = code.replace(
                                            re,
                                            [arrStr, "[", ptrStr, "]"].join(""),
                                        );
                                    }
                                }
                                break;
                            case "scalar":
                                code = code.replace(
                                    re,
                                    "Y" + proc.scalarArgs.indexOf(i),
                                );
                                break;
                            case "index":
                                code = code.replace(re, "index");
                                break;
                            case "shape":
                                code = code.replace(re, "shape");
                                break;
                        }
                    }
                    return [pre.join("\n"), code, post.join("\n")]
                        .join("\n")
                        .trim();
                }

                function typeSummary(dtypes) {
                    var summary = new Array(dtypes.length);
                    var allEqual = true;
                    for (var i = 0; i < dtypes.length; ++i) {
                        var t = dtypes[i];
                        var digits = t.match(/\d+/);
                        if (!digits) {
                            digits = "";
                        } else {
                            digits = digits[0];
                        }
                        if (t.charAt(0) === 0) {
                            summary[i] = "u" + t.charAt(1) + digits;
                        } else {
                            summary[i] = t.charAt(0) + digits;
                        }
                        if (i > 0) {
                            allEqual =
                                allEqual && summary[i] === summary[i - 1];
                        }
                    }
                    if (allEqual) {
                        return summary[0];
                    }
                    return summary.join("");
                }

                //Generates a cwise operator
                function generateCWiseOp(proc, typesig) {
                    //Compute dimension
                    // Arrays get put first in typesig, and there are two entries per array (dtype and order), so this gets the number of dimensions in the first array arg.
                    var dimension =
                        (typesig[1].length -
                            Math.abs(proc.arrayBlockIndices[0])) |
                        0;
                    var orders = new Array(proc.arrayArgs.length);
                    var dtypes = new Array(proc.arrayArgs.length);
                    for (var i = 0; i < proc.arrayArgs.length; ++i) {
                        dtypes[i] = typesig[2 * i];
                        orders[i] = typesig[2 * i + 1];
                    }

                    //Determine where block and loop indices start and end
                    var blockBegin = [],
                        blockEnd = []; // These indices are exposed as blocks
                    var loopBegin = [],
                        loopEnd = []; // These indices are iterated over
                    var loopOrders = []; // orders restricted to the loop indices
                    for (var i = 0; i < proc.arrayArgs.length; ++i) {
                        if (proc.arrayBlockIndices[i] < 0) {
                            loopBegin.push(0);
                            loopEnd.push(dimension);
                            blockBegin.push(dimension);
                            blockEnd.push(
                                dimension + proc.arrayBlockIndices[i],
                            );
                        } else {
                            loopBegin.push(proc.arrayBlockIndices[i]); // Non-negative
                            loopEnd.push(proc.arrayBlockIndices[i] + dimension);
                            blockBegin.push(0);
                            blockEnd.push(proc.arrayBlockIndices[i]);
                        }
                        var newOrder = [];
                        for (var j = 0; j < orders[i].length; j++) {
                            if (
                                loopBegin[i] <= orders[i][j] &&
                                orders[i][j] < loopEnd[i]
                            ) {
                                newOrder.push(orders[i][j] - loopBegin[i]); // If this is a loop index, put it in newOrder, subtracting loopBegin, to make sure that all loopOrders are using a common set of indices.
                            }
                        }
                        loopOrders.push(newOrder);
                    }

                    //First create arguments for procedure
                    var arglist = ["SS"]; // SS is the overall shape over which we iterate
                    var code = ["'use strict'"];
                    var vars = [];

                    for (var j = 0; j < dimension; ++j) {
                        vars.push(["s", j, "=SS[", j, "]"].join("")); // The limits for each dimension.
                    }
                    for (var i = 0; i < proc.arrayArgs.length; ++i) {
                        arglist.push("a" + i); // Actual data array
                        arglist.push("t" + i); // Strides
                        arglist.push("p" + i); // Offset in the array at which the data starts (also used for iterating over the data)

                        for (var j = 0; j < dimension; ++j) {
                            // Unpack the strides into vars for looping
                            vars.push(
                                [
                                    "t",
                                    i,
                                    "p",
                                    j,
                                    "=t",
                                    i,
                                    "[",
                                    loopBegin[i] + j,
                                    "]",
                                ].join(""),
                            );
                        }

                        for (
                            var j = 0;
                            j < Math.abs(proc.arrayBlockIndices[i]);
                            ++j
                        ) {
                            // Unpack the strides into vars for block iteration
                            vars.push(
                                [
                                    "t",
                                    i,
                                    "b",
                                    j,
                                    "=t",
                                    i,
                                    "[",
                                    blockBegin[i] + j,
                                    "]",
                                ].join(""),
                            );
                        }
                    }
                    for (var i = 0; i < proc.scalarArgs.length; ++i) {
                        arglist.push("Y" + i);
                    }
                    if (proc.shapeArgs.length > 0) {
                        vars.push("shape=SS.slice(0)"); // Makes the shape over which we iterate available to the user defined functions (so you can use width/height for example)
                    }
                    if (proc.indexArgs.length > 0) {
                        // Prepare an array to keep track of the (logical) indices, initialized to dimension zeroes.
                        var zeros = new Array(dimension);
                        for (var i = 0; i < dimension; ++i) {
                            zeros[i] = "0";
                        }
                        vars.push(["index=[", zeros.join(","), "]"].join(""));
                    }
                    for (var i = 0; i < proc.offsetArgs.length; ++i) {
                        // Offset arguments used for stencil operations
                        var off_arg = proc.offsetArgs[i];
                        var init_string = [];
                        for (var j = 0; j < off_arg.offset.length; ++j) {
                            if (off_arg.offset[j] === 0) {
                                continue;
                            } else if (off_arg.offset[j] === 1) {
                                init_string.push(
                                    ["t", off_arg.array, "p", j].join(""),
                                );
                            } else {
                                init_string.push(
                                    [
                                        off_arg.offset[j],
                                        "*t",
                                        off_arg.array,
                                        "p",
                                        j,
                                    ].join(""),
                                );
                            }
                        }
                        if (init_string.length === 0) {
                            vars.push("q" + i + "=0");
                        } else {
                            vars.push(
                                ["q", i, "=", init_string.join("+")].join(""),
                            );
                        }
                    }

                    //Prepare this variables
                    var thisVars = uniq(
                        []
                            .concat(proc.pre.thisVars)
                            .concat(proc.body.thisVars)
                            .concat(proc.post.thisVars),
                    );
                    vars = vars.concat(thisVars);
                    if (vars.length > 0) {
                        code.push("var " + vars.join(","));
                    }
                    for (var i = 0; i < proc.arrayArgs.length; ++i) {
                        code.push("p" + i + "|=0");
                    }

                    //Inline prelude
                    if (proc.pre.body.length > 3) {
                        code.push(processBlock(proc.pre, proc, dtypes));
                    }

                    //Process body
                    var body = processBlock(proc.body, proc, dtypes);
                    var matched = countMatches(loopOrders);
                    if (matched < dimension) {
                        code.push(
                            outerFill(matched, loopOrders[0], proc, body),
                        ); // TODO: Rather than passing loopOrders[0], it might be interesting to look at passing an order that represents the majority of the arguments for example.
                    } else {
                        code.push(innerFill(loopOrders[0], proc, body));
                    }

                    //Inline epilog
                    if (proc.post.body.length > 3) {
                        code.push(processBlock(proc.post, proc, dtypes));
                    }

                    if (proc.debug) {
                        console.log(
                            "-----Generated cwise routine for ",
                            typesig,
                            ":\n" + code.join("\n") + "\n----------",
                        );
                    }

                    var loopName = [
                        proc.funcName || "unnamed",
                        "_cwise_loop_",
                        orders[0].join("s"),
                        "m",
                        matched,
                        typeSummary(dtypes),
                    ].join("");
                    var f = new Function(
                        [
                            "function ",
                            loopName,
                            "(",
                            arglist.join(","),
                            "){",
                            code.join("\n"),
                            "} return ",
                            loopName,
                        ].join(""),
                    );
                    return f();
                }
                compile = generateCWiseOp;
            })();

            createThunk = function (proc) {
                var code = ["'use strict'", "var CACHED={}"];
                var vars = [];
                var thunkName = proc.funcName + "_cwise_thunk";

                //Build thunk
                code.push(
                    [
                        "return function ",
                        thunkName,
                        "(",
                        proc.shimArgs.join(","),
                        "){",
                    ].join(""),
                );
                var typesig = [];
                var string_typesig = [];
                var proc_args = [
                    [
                        "array",
                        proc.arrayArgs[0],
                        ".shape.slice(", // Slice shape so that we only retain the shape over which we iterate (which gets passed to the cwise operator as SS).
                        Math.max(0, proc.arrayBlockIndices[0]),
                        proc.arrayBlockIndices[0] < 0
                            ? "," + proc.arrayBlockIndices[0] + ")"
                            : ")",
                    ].join(""),
                ];
                var shapeLengthConditions = [],
                    shapeConditions = [];
                // Process array arguments
                for (var i = 0; i < proc.arrayArgs.length; ++i) {
                    var j = proc.arrayArgs[i];
                    vars.push(
                        [
                            "t",
                            j,
                            "=array",
                            j,
                            ".dtype,",
                            "r",
                            j,
                            "=array",
                            j,
                            ".order",
                        ].join(""),
                    );
                    typesig.push("t" + j);
                    typesig.push("r" + j);
                    string_typesig.push("t" + j);
                    string_typesig.push("r" + j + ".join()");
                    proc_args.push("array" + j + ".data");
                    proc_args.push("array" + j + ".stride");
                    proc_args.push("array" + j + ".offset|0");
                    if (i > 0) {
                        // Gather conditions to check for shape equality (ignoring block indices)
                        shapeLengthConditions.push(
                            "array" +
                                proc.arrayArgs[0] +
                                ".shape.length===array" +
                                j +
                                ".shape.length+" +
                                (Math.abs(proc.arrayBlockIndices[0]) -
                                    Math.abs(proc.arrayBlockIndices[i])),
                        );
                        shapeConditions.push(
                            "array" +
                                proc.arrayArgs[0] +
                                ".shape[shapeIndex+" +
                                Math.max(0, proc.arrayBlockIndices[0]) +
                                "]===array" +
                                j +
                                ".shape[shapeIndex+" +
                                Math.max(0, proc.arrayBlockIndices[i]) +
                                "]",
                        );
                    }
                }
                // Check for shape equality
                if (proc.arrayArgs.length > 1) {
                    code.push(
                        "if (!(" +
                            shapeLengthConditions.join(" && ") +
                            ")) throw new Error('cwise: Arrays do not all have the same dimensionality!')",
                    );
                    code.push(
                        "for(var shapeIndex=array" +
                            proc.arrayArgs[0] +
                            ".shape.length-" +
                            Math.abs(proc.arrayBlockIndices[0]) +
                            "; shapeIndex-->0;) {",
                    );
                    code.push(
                        "if (!(" +
                            shapeConditions.join(" && ") +
                            ")) throw new Error('cwise: Arrays do not all have the same shape!')",
                    );
                    code.push("}");
                }
                // Process scalar arguments
                for (var i = 0; i < proc.scalarArgs.length; ++i) {
                    proc_args.push("scalar" + proc.scalarArgs[i]);
                }
                // Check for cached function (and if not present, generate it)
                vars.push(
                    ["type=[", string_typesig.join(","), "].join()"].join(""),
                );
                vars.push("proc=CACHED[type]");
                code.push("var " + vars.join(","));

                code.push(
                    [
                        "if(!proc){",
                        "CACHED[type]=proc=compile([",
                        typesig.join(","),
                        "])}",
                        "return proc(",
                        proc_args.join(","),
                        ")}",
                    ].join(""),
                );

                if (proc.debug) {
                    console.log(
                        "-----Generated thunk:\n" +
                            code.join("\n") +
                            "\n----------",
                    );
                }

                //Compile thunk
                var thunk = new Function("compile", code.join("\n"));
                return thunk(compile.bind(undefined, proc));
            };
        })();

        var compile;
        (function () {
            function Procedure() {
                this.argTypes = [];
                this.shimArgs = [];
                this.arrayArgs = [];
                this.arrayBlockIndices = [];
                this.scalarArgs = [];
                this.offsetArgs = [];
                this.offsetArgIndex = [];
                this.indexArgs = [];
                this.shapeArgs = [];
                this.funcName = "";
                this.pre = null;
                this.body = null;
                this.post = null;
                this.debug = false;
            }

            function compileCwise(user_args) {
                //Create procedure
                var proc = new Procedure();

                //Parse blocks
                proc.pre = user_args.pre;
                proc.body = user_args.body;
                proc.post = user_args.post;

                //Parse arguments
                var proc_args = user_args.args.slice(0);
                proc.argTypes = proc_args;
                for (var i = 0; i < proc_args.length; ++i) {
                    var arg_type = proc_args[i];
                    if (
                        arg_type === "array" ||
                        (typeof arg_type === "object" && arg_type.blockIndices)
                    ) {
                        proc.argTypes[i] = "array";
                        proc.arrayArgs.push(i);
                        proc.arrayBlockIndices.push(
                            arg_type.blockIndices ? arg_type.blockIndices : 0,
                        );
                        proc.shimArgs.push("array" + i);
                        if (
                            i < proc.pre.args.length &&
                            proc.pre.args[i].count > 0
                        ) {
                            throw new Error(
                                "cwise: pre() block may not reference array args",
                            );
                        }
                        if (
                            i < proc.post.args.length &&
                            proc.post.args[i].count > 0
                        ) {
                            throw new Error(
                                "cwise: post() block may not reference array args",
                            );
                        }
                    } else if (arg_type === "scalar") {
                        proc.scalarArgs.push(i);
                        proc.shimArgs.push("scalar" + i);
                    } else if (arg_type === "index") {
                        proc.indexArgs.push(i);
                        if (
                            i < proc.pre.args.length &&
                            proc.pre.args[i].count > 0
                        ) {
                            throw new Error(
                                "cwise: pre() block may not reference array index",
                            );
                        }
                        if (
                            i < proc.body.args.length &&
                            proc.body.args[i].lvalue
                        ) {
                            throw new Error(
                                "cwise: body() block may not write to array index",
                            );
                        }
                        if (
                            i < proc.post.args.length &&
                            proc.post.args[i].count > 0
                        ) {
                            throw new Error(
                                "cwise: post() block may not reference array index",
                            );
                        }
                    } else if (arg_type === "shape") {
                        proc.shapeArgs.push(i);
                        if (
                            i < proc.pre.args.length &&
                            proc.pre.args[i].lvalue
                        ) {
                            throw new Error(
                                "cwise: pre() block may not write to array shape",
                            );
                        }
                        if (
                            i < proc.body.args.length &&
                            proc.body.args[i].lvalue
                        ) {
                            throw new Error(
                                "cwise: body() block may not write to array shape",
                            );
                        }
                        if (
                            i < proc.post.args.length &&
                            proc.post.args[i].lvalue
                        ) {
                            throw new Error(
                                "cwise: post() block may not write to array shape",
                            );
                        }
                    } else if (
                        typeof arg_type === "object" &&
                        arg_type.offset
                    ) {
                        proc.argTypes[i] = "offset";
                        proc.offsetArgs.push({
                            array: arg_type.array,
                            offset: arg_type.offset,
                        });
                        proc.offsetArgIndex.push(i);
                    } else {
                        throw new Error(
                            "cwise: Unknown argument type " + proc_args[i],
                        );
                    }
                }

                //Make sure at least one array argument was specified
                if (proc.arrayArgs.length <= 0) {
                    throw new Error("cwise: No array arguments specified");
                }

                //Make sure arguments are correct
                if (proc.pre.args.length > proc_args.length) {
                    throw new Error("cwise: Too many arguments in pre() block");
                }
                if (proc.body.args.length > proc_args.length) {
                    throw new Error(
                        "cwise: Too many arguments in body() block",
                    );
                }
                if (proc.post.args.length > proc_args.length) {
                    throw new Error(
                        "cwise: Too many arguments in post() block",
                    );
                }

                //Check debug flag
                proc.debug = !!user_args.printCode || !!user_args.debug;

                //Retrieve name
                proc.funcName = user_args.funcName || "cwise";

                //Read in block size
                proc.blockSize = user_args.blockSize || 64;

                return createThunk(proc);
            }

            compile = compileCwise;
        })();

        var EmptyProc = {
            body: "",
            args: [],
            thisVars: [],
            localVars: [],
        };

        function fixup(x) {
            if (!x) {
                return EmptyProc;
            }
            for (var i = 0; i < x.args.length; ++i) {
                var a = x.args[i];
                if (i === 0) {
                    x.args[i] = {
                        name: a,
                        lvalue: true,
                        rvalue: !!x.rvalue,
                        count: x.count || 1,
                    };
                } else {
                    x.args[i] = {
                        name: a,
                        lvalue: false,
                        rvalue: true,
                        count: 1,
                    };
                }
            }
            if (!x.thisVars) {
                x.thisVars = [];
            }
            if (!x.localVars) {
                x.localVars = [];
            }
            return x;
        }

        function pcompile(user_args) {
            return compile({
                args: user_args.args,
                pre: fixup(user_args.pre),
                body: fixup(user_args.body),
                post: fixup(user_args.proc),
                funcName: user_args.funcName,
            });
        }

        function makeOp(user_args) {
            var args = [];
            for (var i = 0; i < user_args.args.length; ++i) {
                args.push("a" + i);
            }
            var wrapper = new Function(
                "P",
                [
                    "return function ",
                    user_args.funcName,
                    "_ndarrayops(",
                    args.join(","),
                    ") {P(",
                    args.join(","),
                    ");return a0}",
                ].join(""),
            );
            return wrapper(pcompile(user_args));
        }

        var assign_ops = {
            add: "+",
            sub: "-",
            mul: "*",
            div: "/",
            mod: "%",
            band: "&",
            bor: "|",
            bxor: "^",
            lshift: "<<",
            rshift: ">>",
            rrshift: ">>>",
        };
        (function () {
            for (var id in assign_ops) {
                var op = assign_ops[id];
                ops[id] = makeOp({
                    args: ["array", "array", "array"],
                    body: { args: ["a", "b", "c"], body: "a=b" + op + "c" },
                    funcName: id,
                });
                ops[id + "eq"] = makeOp({
                    args: ["array", "array"],
                    body: { args: ["a", "b"], body: "a" + op + "=b" },
                    rvalue: true,
                    funcName: id + "eq",
                });
                ops[id + "s"] = makeOp({
                    args: ["array", "array", "scalar"],
                    body: { args: ["a", "b", "s"], body: "a=b" + op + "s" },
                    funcName: id + "s",
                });
                ops[id + "seq"] = makeOp({
                    args: ["array", "scalar"],
                    body: { args: ["a", "s"], body: "a" + op + "=s" },
                    rvalue: true,
                    funcName: id + "seq",
                });
            }
        })();

        var unary_ops = {
            not: "!",
            bnot: "~",
            neg: "-",
            recip: "1.0/",
        };
        (function () {
            for (var id in unary_ops) {
                var op = unary_ops[id];
                ops[id] = makeOp({
                    args: ["array", "array"],
                    body: { args: ["a", "b"], body: "a=" + op + "b" },
                    funcName: id,
                });
                ops[id + "eq"] = makeOp({
                    args: ["array"],
                    body: { args: ["a"], body: "a=" + op + "a" },
                    rvalue: true,
                    count: 2,
                    funcName: id + "eq",
                });
            }
        })();

        var binary_ops = {
            and: "&&",
            or: "||",
            eq: "===",
            neq: "!==",
            lt: "<",
            gt: ">",
            leq: "<=",
            geq: ">=",
        };
        (function () {
            for (var id in binary_ops) {
                var op = binary_ops[id];
                ops[id] = makeOp({
                    args: ["array", "array", "array"],
                    body: { args: ["a", "b", "c"], body: "a=b" + op + "c" },
                    funcName: id,
                });
                ops[id + "s"] = makeOp({
                    args: ["array", "array", "scalar"],
                    body: { args: ["a", "b", "s"], body: "a=b" + op + "s" },
                    funcName: id + "s",
                });
                ops[id + "eq"] = makeOp({
                    args: ["array", "array"],
                    body: { args: ["a", "b"], body: "a=a" + op + "b" },
                    rvalue: true,
                    count: 2,
                    funcName: id + "eq",
                });
                ops[id + "seq"] = makeOp({
                    args: ["array", "scalar"],
                    body: { args: ["a", "s"], body: "a=a" + op + "s" },
                    rvalue: true,
                    count: 2,
                    funcName: id + "seq",
                });
            }
        })();

        var math_unary = [
            "abs",
            "acos",
            "asin",
            "atan",
            "ceil",
            "cos",
            "exp",
            "floor",
            "log",
            "round",
            "sin",
            "sqrt",
            "tan",
        ];
        (function () {
            for (var i = 0; i < math_unary.length; ++i) {
                var f = math_unary[i];
                ops[f] = makeOp({
                    args: ["array", "array"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b"],
                        body: "a=this_f(b)",
                        thisVars: ["this_f"],
                    },
                    funcName: f,
                });
                ops[f + "eq"] = makeOp({
                    args: ["array"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a"],
                        body: "a=this_f(a)",
                        thisVars: ["this_f"],
                    },
                    rvalue: true,
                    count: 2,
                    funcName: f + "eq",
                });
            }
        })();

        var math_comm = ["max", "min", "atan2", "pow"];
        (function () {
            for (var i = 0; i < math_comm.length; ++i) {
                var f = math_comm[i];
                ops[f] = makeOp({
                    args: ["array", "array", "array"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b", "c"],
                        body: "a=this_f(b,c)",
                        thisVars: ["this_f"],
                    },
                    funcName: f,
                });
                ops[f + "s"] = makeOp({
                    args: ["array", "array", "scalar"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b", "c"],
                        body: "a=this_f(b,c)",
                        thisVars: ["this_f"],
                    },
                    funcName: f + "s",
                });
                ops[f + "eq"] = makeOp({
                    args: ["array", "array"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b"],
                        body: "a=this_f(a,b)",
                        thisVars: ["this_f"],
                    },
                    rvalue: true,
                    count: 2,
                    funcName: f + "eq",
                });
                ops[f + "seq"] = makeOp({
                    args: ["array", "scalar"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b"],
                        body: "a=this_f(a,b)",
                        thisVars: ["this_f"],
                    },
                    rvalue: true,
                    count: 2,
                    funcName: f + "seq",
                });
            }
        })();

        var math_noncomm = ["atan2", "pow"];
        (function () {
            for (var i = 0; i < math_noncomm.length; ++i) {
                var f = math_noncomm[i];
                ops[f + "op"] = makeOp({
                    args: ["array", "array", "array"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b", "c"],
                        body: "a=this_f(c,b)",
                        thisVars: ["this_f"],
                    },
                    funcName: f + "op",
                });
                ops[f + "ops"] = makeOp({
                    args: ["array", "array", "scalar"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b", "c"],
                        body: "a=this_f(c,b)",
                        thisVars: ["this_f"],
                    },
                    funcName: f + "ops",
                });
                ops[f + "opeq"] = makeOp({
                    args: ["array", "array"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b"],
                        body: "a=this_f(b,a)",
                        thisVars: ["this_f"],
                    },
                    rvalue: true,
                    count: 2,
                    funcName: f + "opeq",
                });
                ops[f + "opseq"] = makeOp({
                    args: ["array", "scalar"],
                    pre: {
                        args: [],
                        body: "this_f=Math." + f,
                        thisVars: ["this_f"],
                    },
                    body: {
                        args: ["a", "b"],
                        body: "a=this_f(b,a)",
                        thisVars: ["this_f"],
                    },
                    rvalue: true,
                    count: 2,
                    funcName: f + "opseq",
                });
            }
        })();

        ops.any = compile({
            args: ["array"],
            pre: EmptyProc,
            body: {
                args: [{ name: "a", lvalue: false, rvalue: true, count: 1 }],
                body: "if(a){return true}",
                localVars: [],
                thisVars: [],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: [],
                body: "return false",
            },
            funcName: "any",
        });

        ops.all = compile({
            args: ["array"],
            pre: EmptyProc,
            body: {
                args: [{ name: "x", lvalue: false, rvalue: true, count: 1 }],
                body: "if(!x){return false}",
                localVars: [],
                thisVars: [],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: [],
                body: "return true",
            },
            funcName: "all",
        });

        ops.sum = compile({
            args: ["array"],
            pre: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "this_s=0",
            },
            body: {
                args: [{ name: "a", lvalue: false, rvalue: true, count: 1 }],
                body: "this_s+=a",
                localVars: [],
                thisVars: ["this_s"],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "return this_s",
            },
            funcName: "sum",
        });

        ops.prod = compile({
            args: ["array"],
            pre: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "this_s=1",
            },
            body: {
                args: [{ name: "a", lvalue: false, rvalue: true, count: 1 }],
                body: "this_s*=a",
                localVars: [],
                thisVars: ["this_s"],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "return this_s",
            },
            funcName: "prod",
        });

        ops.norm2squared = compile({
            args: ["array"],
            pre: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "this_s=0",
            },
            body: {
                args: [{ name: "a", lvalue: false, rvalue: true, count: 2 }],
                body: "this_s+=a*a",
                localVars: [],
                thisVars: ["this_s"],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "return this_s",
            },
            funcName: "norm2squared",
        });

        ops.norm2 = compile({
            args: ["array"],
            pre: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "this_s=0",
            },
            body: {
                args: [{ name: "a", lvalue: false, rvalue: true, count: 2 }],
                body: "this_s+=a*a",
                localVars: [],
                thisVars: ["this_s"],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "return Math.sqrt(this_s)",
            },
            funcName: "norm2",
        });

        ops.norminf = compile({
            args: ["array"],
            pre: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "this_s=0",
            },
            body: {
                args: [{ name: "a", lvalue: false, rvalue: true, count: 4 }],
                body: "if(-a>this_s){this_s=-a}else if(a>this_s){this_s=a}",
                localVars: [],
                thisVars: ["this_s"],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "return this_s",
            },
            funcName: "norminf",
        });

        ops.norm1 = compile({
            args: ["array"],
            pre: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "this_s=0",
            },
            body: {
                args: [{ name: "a", lvalue: false, rvalue: true, count: 3 }],
                body: "this_s+=a<0?-a:a",
                localVars: [],
                thisVars: ["this_s"],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: ["this_s"],
                body: "return this_s",
            },
            funcName: "norm1",
        });

        ops.sup = compile({
            args: ["array"],
            pre: {
                body: "this_h=-Infinity",
                args: [],
                thisVars: ["this_h"],
                localVars: [],
            },
            body: {
                body: "if(_inline_1_arg0_>this_h)this_h=_inline_1_arg0_",
                args: [
                    {
                        name: "_inline_1_arg0_",
                        lvalue: false,
                        rvalue: true,
                        count: 2,
                    },
                ],
                thisVars: ["this_h"],
                localVars: [],
            },
            post: {
                body: "return this_h",
                args: [],
                thisVars: ["this_h"],
                localVars: [],
            },
        });

        ops.inf = compile({
            args: ["array"],
            pre: {
                body: "this_h=Infinity",
                args: [],
                thisVars: ["this_h"],
                localVars: [],
            },
            body: {
                body: "if(_inline_1_arg0_<this_h)this_h=_inline_1_arg0_",
                args: [
                    {
                        name: "_inline_1_arg0_",
                        lvalue: false,
                        rvalue: true,
                        count: 2,
                    },
                ],
                thisVars: ["this_h"],
                localVars: [],
            },
            post: {
                body: "return this_h",
                args: [],
                thisVars: ["this_h"],
                localVars: [],
            },
        });

        ops.argmin = compile({
            args: ["index", "array", "shape"],
            pre: {
                body: "{this_v=Infinity;this_i=_inline_0_arg2_.slice(0)}",
                args: [
                    {
                        name: "_inline_0_arg0_",
                        lvalue: false,
                        rvalue: false,
                        count: 0,
                    },
                    {
                        name: "_inline_0_arg1_",
                        lvalue: false,
                        rvalue: false,
                        count: 0,
                    },
                    {
                        name: "_inline_0_arg2_",
                        lvalue: false,
                        rvalue: true,
                        count: 1,
                    },
                ],
                thisVars: ["this_i", "this_v"],
                localVars: [],
            },
            body: {
                body: "{if(_inline_1_arg1_<this_v){this_v=_inline_1_arg1_;for(var _inline_1_k=0;_inline_1_k<_inline_1_arg0_.length;++_inline_1_k){this_i[_inline_1_k]=_inline_1_arg0_[_inline_1_k]}}}",
                args: [
                    {
                        name: "_inline_1_arg0_",
                        lvalue: false,
                        rvalue: true,
                        count: 2,
                    },
                    {
                        name: "_inline_1_arg1_",
                        lvalue: false,
                        rvalue: true,
                        count: 2,
                    },
                ],
                thisVars: ["this_i", "this_v"],
                localVars: ["_inline_1_k"],
            },
            post: {
                body: "{return this_i}",
                args: [],
                thisVars: ["this_i"],
                localVars: [],
            },
        });

        ops.argmax = compile({
            args: ["index", "array", "shape"],
            pre: {
                body: "{this_v=-Infinity;this_i=_inline_0_arg2_.slice(0)}",
                args: [
                    {
                        name: "_inline_0_arg0_",
                        lvalue: false,
                        rvalue: false,
                        count: 0,
                    },
                    {
                        name: "_inline_0_arg1_",
                        lvalue: false,
                        rvalue: false,
                        count: 0,
                    },
                    {
                        name: "_inline_0_arg2_",
                        lvalue: false,
                        rvalue: true,
                        count: 1,
                    },
                ],
                thisVars: ["this_i", "this_v"],
                localVars: [],
            },
            body: {
                body: "{if(_inline_1_arg1_>this_v){this_v=_inline_1_arg1_;for(var _inline_1_k=0;_inline_1_k<_inline_1_arg0_.length;++_inline_1_k){this_i[_inline_1_k]=_inline_1_arg0_[_inline_1_k]}}}",
                args: [
                    {
                        name: "_inline_1_arg0_",
                        lvalue: false,
                        rvalue: true,
                        count: 2,
                    },
                    {
                        name: "_inline_1_arg1_",
                        lvalue: false,
                        rvalue: true,
                        count: 2,
                    },
                ],
                thisVars: ["this_i", "this_v"],
                localVars: ["_inline_1_k"],
            },
            post: {
                body: "{return this_i}",
                args: [],
                thisVars: ["this_i"],
                localVars: [],
            },
        });

        ops.random = makeOp({
            args: ["array"],
            pre: { args: [], body: "this_f=Math.random", thisVars: ["this_f"] },
            body: { args: ["a"], body: "a=this_f()", thisVars: ["this_f"] },
            funcName: "random",
        });

        ops.assign = makeOp({
            args: ["array", "array"],
            body: { args: ["a", "b"], body: "a=b" },
            funcName: "assign",
        });

        ops.assigns = makeOp({
            args: ["array", "scalar"],
            body: { args: ["a", "b"], body: "a=b" },
            funcName: "assigns",
        });

        ops.equals = compile({
            args: ["array", "array"],
            pre: EmptyProc,
            body: {
                args: [
                    { name: "x", lvalue: false, rvalue: true, count: 1 },
                    { name: "y", lvalue: false, rvalue: true, count: 1 },
                ],
                body: "if(x!==y){return false}",
                localVars: [],
                thisVars: [],
            },
            post: {
                args: [],
                localVars: [],
                thisVars: [],
                body: "return true",
            },
            funcName: "equals",
        });
    })();

    var solve;
    (function () {
        var pool = {} as any;
        (function () {
            var bits = {} as any;
            (function () {
                //Number of bits in an integer
                var INT_BITS = 32;

                //Constants
                bits.INT_BITS = INT_BITS;
                bits.INT_MAX = 0x7fffffff;
                bits.INT_MIN = -1 << (INT_BITS - 1);

                //Computes absolute value of integer
                bits.abs = function (v) {
                    var mask = v >> (INT_BITS - 1);
                    return (v ^ mask) - mask;
                };

                //Computes minimum of integers x and y
                bits.min = function (x, y) {
                    return y ^ ((x ^ y) & -(x < y));
                };

                //Computes maximum of integers x and y
                bits.max = function (x, y) {
                    return x ^ ((x ^ y) & -(x < y));
                };

                //Checks if a number is a power of two
                bits.isPow2 = function (v) {
                    return !(v & (v - 1)) && !!v;
                };

                //Computes log base 2 of v
                bits.log2 = function (v: number) {
                    var r, shift;
                    //@ts-ignore
                    r = (v > 0xffff) << 4;
                    v >>>= r;
                    //@ts-ignore
                    shift = (v > 0xff) << 3;
                    v >>>= shift;
                    r |= shift;
                    //@ts-ignore
                    shift = (v > 0xf) << 2;
                    v >>>= shift;
                    r |= shift;
                    //@ts-ignore
                    shift = (v > 0x3) << 1;
                    v >>>= shift;
                    r |= shift;
                    return r | (v >> 1);
                };

                //Computes log base 10 of v
                bits.log10 = function (v) {
                    return v >= 1000000000
                        ? 9
                        : v >= 100000000
                        ? 8
                        : v >= 10000000
                        ? 7
                        : v >= 1000000
                        ? 6
                        : v >= 100000
                        ? 5
                        : v >= 10000
                        ? 4
                        : v >= 1000
                        ? 3
                        : v >= 100
                        ? 2
                        : v >= 10
                        ? 1
                        : 0;
                };

                //Counts number of bits
                bits.popCount = function (v) {
                    v = v - ((v >>> 1) & 0x55555555);
                    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
                    return (((v + (v >>> 4)) & 0xf0f0f0f) * 0x1010101) >>> 24;
                };

                //Counts number of trailing zeros
                function countTrailingZeros(v) {
                    var c = 32;
                    v &= -v;
                    if (v) c--;
                    if (v & 0x0000ffff) c -= 16;
                    if (v & 0x00ff00ff) c -= 8;
                    if (v & 0x0f0f0f0f) c -= 4;
                    if (v & 0x33333333) c -= 2;
                    if (v & 0x55555555) c -= 1;
                    return c;
                }
                bits.countTrailingZeros = countTrailingZeros;

                //Rounds to next power of 2
                bits.nextPow2 = function (v) {
                    v += v === 0;
                    --v;
                    v |= v >>> 1;
                    v |= v >>> 2;
                    v |= v >>> 4;
                    v |= v >>> 8;
                    v |= v >>> 16;
                    return v + 1;
                };

                //Rounds down to previous power of 2
                bits.prevPow2 = function (v) {
                    v |= v >>> 1;
                    v |= v >>> 2;
                    v |= v >>> 4;
                    v |= v >>> 8;
                    v |= v >>> 16;
                    return v - (v >>> 1);
                };

                //Computes parity of word
                bits.parity = function (v) {
                    v ^= v >>> 16;
                    v ^= v >>> 8;
                    v ^= v >>> 4;
                    v &= 0xf;
                    return (0x6996 >>> v) & 1;
                };

                var REVERSE_TABLE = new Array(256);

                (function (tab) {
                    for (var i = 0; i < 256; ++i) {
                        var v = i,
                            r = i,
                            s = 7;
                        for (v >>>= 1; v; v >>>= 1) {
                            r <<= 1;
                            r |= v & 1;
                            --s;
                        }
                        tab[i] = (r << s) & 0xff;
                    }
                })(REVERSE_TABLE);

                //Reverse bits in a 32 bit word
                bits.reverse = function (v) {
                    return (
                        (REVERSE_TABLE[v & 0xff] << 24) |
                        (REVERSE_TABLE[(v >>> 8) & 0xff] << 16) |
                        (REVERSE_TABLE[(v >>> 16) & 0xff] << 8) |
                        REVERSE_TABLE[(v >>> 24) & 0xff]
                    );
                };

                //Interleave bits of 2 coordinates with 16 bits.  Useful for fast quadtree codes
                bits.interleave2 = function (x, y) {
                    x &= 0xffff;
                    x = (x | (x << 8)) & 0x00ff00ff;
                    x = (x | (x << 4)) & 0x0f0f0f0f;
                    x = (x | (x << 2)) & 0x33333333;
                    x = (x | (x << 1)) & 0x55555555;

                    y &= 0xffff;
                    y = (y | (y << 8)) & 0x00ff00ff;
                    y = (y | (y << 4)) & 0x0f0f0f0f;
                    y = (y | (y << 2)) & 0x33333333;
                    y = (y | (y << 1)) & 0x55555555;

                    return x | (y << 1);
                };

                //Extracts the nth interleaved component
                bits.deinterleave2 = function (v, n) {
                    v = (v >>> n) & 0x55555555;
                    v = (v | (v >>> 1)) & 0x33333333;
                    v = (v | (v >>> 2)) & 0x0f0f0f0f;
                    v = (v | (v >>> 4)) & 0x00ff00ff;
                    v = (v | (v >>> 16)) & 0x000ffff;
                    return (v << 16) >> 16;
                };

                //Interleave bits of 3 coordinates, each with 10 bits.  Useful for fast octree codes
                bits.interleave3 = function (x, y, z) {
                    x &= 0x3ff;
                    x = (x | (x << 16)) & 4278190335;
                    x = (x | (x << 8)) & 251719695;
                    x = (x | (x << 4)) & 3272356035;
                    x = (x | (x << 2)) & 1227133513;

                    y &= 0x3ff;
                    y = (y | (y << 16)) & 4278190335;
                    y = (y | (y << 8)) & 251719695;
                    y = (y | (y << 4)) & 3272356035;
                    y = (y | (y << 2)) & 1227133513;
                    x |= y << 1;

                    z &= 0x3ff;
                    z = (z | (z << 16)) & 4278190335;
                    z = (z | (z << 8)) & 251719695;
                    z = (z | (z << 4)) & 3272356035;
                    z = (z | (z << 2)) & 1227133513;

                    return x | (z << 2);
                };

                //Extracts nth interleaved component of a 3-tuple
                bits.deinterleave3 = function (v, n) {
                    v = (v >>> n) & 1227133513;
                    v = (v | (v >>> 2)) & 3272356035;
                    v = (v | (v >>> 4)) & 251719695;
                    v = (v | (v >>> 8)) & 4278190335;
                    v = (v | (v >>> 16)) & 0x3ff;
                    return (v << 22) >> 22;
                };

                //Computes next combination in colexicographic order (this is mistakenly called nextPermutation on the bit twiddling hacks page)
                bits.nextCombination = function (v) {
                    var t = v | (v - 1);
                    return (
                        (t + 1) |
                        (((~t & -~t) - 1) >>> (countTrailingZeros(v) + 1))
                    );
                };
            })();

            var dup;
            (function () {
                function dupe_array(count, value, i) {
                    var c = count[i] | 0;
                    if (c <= 0) {
                        return [];
                    }
                    var result = new Array(c),
                        j;
                    if (i === count.length - 1) {
                        for (j = 0; j < c; ++j) {
                            result[j] = value;
                        }
                    } else {
                        for (j = 0; j < c; ++j) {
                            result[j] = dupe_array(count, value, i + 1);
                        }
                    }
                    return result;
                }
                function dupe_number(count, value) {
                    var result, i;
                    result = new Array(count);
                    for (i = 0; i < count; ++i) {
                        result[i] = value;
                    }
                    return result;
                }
                function dupe(count, value) {
                    if (typeof value === "undefined") {
                        value = 0;
                    }
                    switch (typeof count) {
                        case "number":
                            if (count > 0) {
                                return dupe_number(count | 0, value);
                            }
                            break;
                        case "object":
                            if (typeof count.length === "number") {
                                return dupe_array(count, value, 0);
                            }
                            break;
                    }
                    return [];
                }
                dup = dupe;
            })();

            var hasUint8C = typeof Uint8ClampedArray !== "undefined";
            var hasBigUint64 = typeof BigUint64Array !== "undefined";
            var hasBigInt64 = typeof BigInt64Array !== "undefined";
            var POOL = {
                UINT8: dup([32, 0]),
                UINT16: dup([32, 0]),
                UINT32: dup([32, 0]),
                BIGUINT64: dup([32, 0]),
                INT8: dup([32, 0]),
                INT16: dup([32, 0]),
                INT32: dup([32, 0]),
                BIGINT64: dup([32, 0]),
                FLOAT: dup([32, 0]),
                DOUBLE: dup([32, 0]),
                DATA: dup([32, 0]),
                UINT8C: dup([32, 0]),
                BUFFER: dup([32, 0]),
            };

            //Upgrade pool
            if (!POOL.UINT8C) {
                POOL.UINT8C = dup([32, 0]);
            }
            if (!POOL.BIGUINT64) {
                POOL.BIGUINT64 = dup([32, 0]);
            }
            if (!POOL.BIGINT64) {
                POOL.BIGINT64 = dup([32, 0]);
            }
            if (!POOL.BUFFER) {
                POOL.BUFFER = dup([32, 0]);
            }

            var DATA = POOL.DATA;

            pool.free = function free(array) {
                if (
                    Object.prototype.toString.call(array) !==
                    "[object ArrayBuffer]"
                ) {
                    array = array.buffer;
                }
                if (!array) {
                    return;
                }
                var n = array.length || array.byteLength;
                var log_n = bits.log2(n) | 0;
                DATA[log_n].push(array);
            };

            function freeArrayBuffer(buffer) {
                if (!buffer) {
                    return;
                }
                var n = buffer.length || buffer.byteLength;
                var log_n = bits.log2(n);
                DATA[log_n].push(buffer);
            }

            function freeTypedArray(array) {
                freeArrayBuffer(array.buffer);
            }

            pool.freeUint8 =
                pool.freeUint16 =
                pool.freeUint32 =
                pool.freeBigUint64 =
                pool.freeInt8 =
                pool.freeInt16 =
                pool.freeInt32 =
                pool.freeBigInt64 =
                pool.freeFloat32 =
                pool.freeFloat =
                pool.freeFloat64 =
                pool.freeDouble =
                pool.freeUint8Clamped =
                pool.freeDataView =
                    freeTypedArray;

            pool.freeArrayBuffer = freeArrayBuffer;

            pool.malloc = function malloc(n, dtype) {
                if (dtype === undefined || dtype === "arraybuffer") {
                    return mallocArrayBuffer(n);
                } else {
                    switch (dtype) {
                        case "uint8":
                            return mallocUint8(n);
                        case "uint16":
                            return mallocUint16(n);
                        case "uint32":
                            return mallocUint32(n);
                        case "int8":
                            return mallocInt8(n);
                        case "int16":
                            return mallocInt16(n);
                        case "int32":
                            return mallocInt32(n);
                        case "float":
                        case "float32":
                            return mallocFloat(n);
                        case "double":
                        case "float64":
                            return mallocDouble(n);
                        case "uint8_clamped":
                            return mallocUint8Clamped(n);
                        case "bigint64":
                            return mallocBigInt64(n);
                        case "biguint64":
                            return mallocBigUint64(n);
                        case "data":
                        case "dataview":
                            return mallocDataView(n);

                        default:
                            return null;
                    }
                }
                return null;
            };

            function mallocArrayBuffer(n) {
                var n = bits.nextPow2(n);
                var log_n = bits.log2(n);
                var d = DATA[log_n];
                if (d.length > 0) {
                    return d.pop();
                }
                return new ArrayBuffer(n);
            }
            pool.mallocArrayBuffer = mallocArrayBuffer;

            function mallocUint8(n) {
                return new Uint8Array(mallocArrayBuffer(n), 0, n);
            }
            pool.mallocUint8 = mallocUint8;

            function mallocUint16(n) {
                return new Uint16Array(mallocArrayBuffer(2 * n), 0, n);
            }
            pool.mallocUint16 = mallocUint16;

            function mallocUint32(n) {
                return new Uint32Array(mallocArrayBuffer(4 * n), 0, n);
            }
            pool.mallocUint32 = mallocUint32;

            function mallocInt8(n) {
                return new Int8Array(mallocArrayBuffer(n), 0, n);
            }
            pool.mallocInt8 = mallocInt8;

            function mallocInt16(n) {
                return new Int16Array(mallocArrayBuffer(2 * n), 0, n);
            }
            pool.mallocInt16 = mallocInt16;

            function mallocInt32(n) {
                return new Int32Array(mallocArrayBuffer(4 * n), 0, n);
            }
            pool.mallocInt32 = mallocInt32;

            function mallocFloat(n) {
                return new Float32Array(mallocArrayBuffer(4 * n), 0, n);
            }
            pool.mallocFloat32 = pool.mallocFloat = mallocFloat;

            function mallocDouble(n) {
                return new Float64Array(mallocArrayBuffer(8 * n), 0, n);
            }
            pool.mallocFloat64 = pool.mallocDouble = mallocDouble;

            function mallocUint8Clamped(n) {
                if (hasUint8C) {
                    return new Uint8ClampedArray(mallocArrayBuffer(n), 0, n);
                } else {
                    return mallocUint8(n);
                }
            }
            pool.mallocUint8Clamped = mallocUint8Clamped;

            function mallocBigUint64(n) {
                if (hasBigUint64) {
                    return new BigUint64Array(mallocArrayBuffer(8 * n), 0, n);
                } else {
                    return null;
                }
            }
            pool.mallocBigUint64 = mallocBigUint64;

            function mallocBigInt64(n) {
                if (hasBigInt64) {
                    return new BigInt64Array(mallocArrayBuffer(8 * n), 0, n);
                } else {
                    return null;
                }
            }
            pool.mallocBigInt64 = mallocBigInt64;

            function mallocDataView(n) {
                return new DataView(mallocArrayBuffer(n), 0, n);
            }
            pool.mallocDataView = mallocDataView;

            pool.clearCache = function clearCache() {
                for (var i = 0; i < 32; ++i) {
                    POOL.UINT8[i].length = 0;
                    POOL.UINT16[i].length = 0;
                    POOL.UINT32[i].length = 0;
                    POOL.INT8[i].length = 0;
                    POOL.INT16[i].length = 0;
                    POOL.INT32[i].length = 0;
                    POOL.FLOAT[i].length = 0;
                    POOL.DOUBLE[i].length = 0;
                    POOL.BIGUINT64[i].length = 0;
                    POOL.BIGINT64[i].length = 0;
                    POOL.UINT8C[i].length = 0;
                    DATA[i].length = 0;
                }
            };
        })();

        var scratch = {} as any;
        (function () {
            function clone(array) {
                var dtype = array.dtype;
                if (dtype === "generic" || dtype === "array") {
                    dtype = "double";
                }
                var data = pool.malloc(array.size, dtype);
                var result = ndarray(data, array.shape);
                ops.assign(result, array);
                return result;
            }
            scratch.clone = clone;

            function malloc(shape, dtype) {
                if (!dtype) {
                    dtype = "double";
                }
                var sz = 1;
                var stride = new Array(shape.length);
                for (var i = shape.length - 1; i >= 0; --i) {
                    stride[i] = sz;
                    sz *= shape[i];
                }
                return ndarray(pool.malloc(sz, dtype), shape, stride, 0);
            }
            scratch.malloc = malloc;

            function free(array) {
                if (array.dtype === "generic" || array.dtype === "array") {
                    return;
                }
                pool.free(array.data);
            }
            scratch.free = free;

            function zeros(shape, dtype) {
                if (!dtype) {
                    dtype = "double";
                }

                var sz = 1;
                var stride = new Array(shape.length);
                for (var i = shape.length - 1; i >= 0; --i) {
                    stride[i] = sz;
                    sz *= shape[i];
                }
                var buf = pool.malloc(sz, dtype);
                for (var i = 0; i < sz; ++i) {
                    buf[i] = 0;
                }
                return ndarray(buf, shape, stride, 0);
            }
            scratch.zeros = zeros;

            function ones(shape, dtype) {
                if (!dtype) {
                    dtype = "double";
                }

                var sz = 1;
                var stride = new Array(shape.length);
                for (var i = shape.length - 1; i >= 0; --i) {
                    stride[i] = sz;
                    sz *= shape[i];
                }
                var buf = pool.malloc(sz, dtype);
                for (var i = 0; i < sz; ++i) {
                    buf[i] = 1;
                }
                return ndarray(buf, shape, stride, 0);
            }
            scratch.ones = ones;

            function eye(shape, dtype) {
                var i, offset;
                if (!dtype) {
                    dtype = "double";
                }

                var sz = 1;
                var stride = new Array(shape.length);
                for (i = shape.length - 1; i >= 0; --i) {
                    stride[i] = sz;
                    sz *= shape[i];
                }
                var buf = pool.malloc(sz, dtype);
                for (i = 0; i < sz; ++i) {
                    buf[i] = 0;
                }
                var mindim = Infinity;
                var offsum = 0;
                for (i = shape.length - 1; i >= 0; i--) {
                    offsum += stride[i];
                    mindim = Math.min(mindim, shape[i]);
                }
                for (i = 0, offset = 0; i < mindim; i++, offset += offsum) {
                    buf[offset] = 1;
                }
                return ndarray(buf, shape, stride, 0);
            }
            scratch.eye = eye;
        })();

        var lusolve;
        (function () {
            lusolve = function (L, U, B, X, Y) {
                var m = L.shape[0],
                    n = L.shape[1],
                    freeY = false;
                if (U.dimension === 1) {
                    Y = X;
                    X = B;
                    B = U;
                    U = L;
                }
                if (!X) X = scratch.malloc([m]);
                if (!Y) {
                    Y = scratch.malloc([m]);
                    freeY = true;
                }

                // LY = B, solve for Y
                for (var y = 0; y < n; y++) {
                    var c = 0;
                    for (var x = 0; x < y; x++) {
                        c += L.get(x, y) * Y.get(x);
                    }
                    Y.set(y, (B.get(y) - c) / L.get(y, y));
                }

                //UX = Y, solve for X
                for (var y = n - 1; y >= 0; y--) {
                    var c = 0;
                    for (var x = n - 1; x > y; x--) {
                        c += U.get(x, y) * X.get(x);
                    }
                    X.set(y, Y.get(y) - c);
                }

                if (freeY) scratch.free(Y);

                return X;
            };
        })();

        var crout;
        (function () {
            crout = function (A, L, U) {
                var m = A.shape[0];
                var n = A.shape[1];
                if (m !== n) return false; // non-square
                if (L && !U) U = L;

                // diagonalize U
                for (var i = 0; i < n; i++) {
                    U.set(i, i, 1);
                }

                for (var j = 0; j < n; j++) {
                    for (var i = j; i < n; i++) {
                        var sum = 0;
                        for (var k = 0; k < j; k++) {
                            sum += L.get(k, i) * U.get(j, k);
                        }
                        L.set(j, i, A.get(j, i) - sum);
                    }

                    var denom = L.get(j, j);
                    if (denom === 0) return false;

                    for (var i = j + 1; i < n; i++) {
                        var sum = 0;
                        for (var k = 0; k < j; k++) {
                            sum += L.get(k, j) * U.get(i, k);
                        }
                        U.set(i, j, (A.get(i, j) - sum) / denom);
                    }
                }
                return true;
            };
        })();

        //TODO: Switch this to LUP eventually for better numerical stability and to support large matrices
        function denseGeneralSolve(m, X, A, B) {
            var L = scratch.malloc([m, m]);
            var ok = crout(A, L, L);
            if (!ok) {
                scratch.free(L);
                return false;
            }
            var Y = scratch.malloc([m]);
            var res = lusolve(L, L, B, X, Y);
            scratch.free(Y);
            scratch.free(L);
            return !!res;
        }
        solve = function (X, A, B) {
            //Validate inputs
            if (A.dimension !== 2)
                throw new Error("not a 2-dimensional matrix");
            var m = A.shape[0],
                n = A.shape[1];
            if (m !== n) throw new Error("not a square matrix: " + m + "x" + n);
            if (B.dimension !== 1) throw new Error("B is not a vector");
            if (B.shape[0] !== m) throw new Error("B has an invalid length");
            if (X.dimension !== 1) throw new Error("X is not a vector");
            if (X.shape[0] !== m) throw new Error("X has an invalid length");

            //TODO: Implement other solvers based on the format of M
            //
            //  Would be nice to have:
            //
            //      * QR factorization for rectangular matrices
            //      * Cholesky/LDL solver for positive semidefinite matrices
            //      * Sparse solvers
            //

            return denseGeneralSolve(m, X, A, B);
        };
    })();
    /////////////////////HELPERS/////////////////////

    function vertexError(vertex, quadratic): number {
        var xformed = new Array(4);
        vec4.transformMat4(xformed, vertex, quadratic);
        return vec4.dot(vertex, xformed);
    }

    function optimalPosition(v1, v2) {
        var q1 = v1.error;
        var q2 = v2.error;
        var costMatrix = ndarray(new Float32Array(4 * 4), [4, 4]);
        ops.add(costMatrix, q1, q2);
        var mat4Cost = Array.from(costMatrix.data);
        var optimal = ndarray(new Float32Array(4));
        var toInvert = costMatrix;
        toInvert.set(0, 3, 0);
        toInvert.set(1, 3, 0);
        toInvert.set(2, 3, 0);
        toInvert.set(3, 3, 1);
        var solved = solve(optimal, toInvert, ndarray([0, 0, 0, 1]));

        if (!solved) {
            var v1Homogenous = Array.from(v1.position);
            v1Homogenous.push(1);
            var v2Homogenous = Array.from(v2.position);
            v2Homogenous.push(1);
            var midpoint = vec3.add(new Array(3), v1.position, v2.position);
            vec3.scale(midpoint, midpoint, 0.5);
            midpoint.push(1);
            var v1Error = vertexError(v1Homogenous, mat4Cost);
            var v2Error = vertexError(v2Homogenous, mat4Cost);
            var midpointError = vertexError(midpoint, mat4Cost);
            var minimum = Math.min(...[v1Error, v2Error, midpointError]);
            if (v1Error == minimum) {
                optimal = v1Homogenous;
            } else if (v2Error == minimum) {
                optimal = v2Homogenous;
            } else {
                optimal = midpoint;
            }
        } else {
            optimal = optimal.data;
        }

        var error = vertexError(optimal, mat4Cost);
        return { vertex: optimal.slice(0, 3), color: v1.color, error: error };
    }

    function meshSimplify(data, threshold = 0) {
        const positions = data.positions;
        const colors = data.colors || undefined;
        const cells = removeDegenerates(data.cells);
        const faceNormals = normals(cells, positions);

        var n = positions.length;
        var vertices = positions.map(function (p, i) {
            return {
                position: p,
                color: colors !== undefined ? colors[i] : undefined,
                index: i,
                pairs: [],
                error: ndarray(new Float32Array(4 * 4).fill(0), [4, 4]),
            };
        });

        cells.map(function (cell) {
            for (var i = 0; i < 2; i++) {
                var j = (i + 1) % 3;
                var v1 = cell[i];
                var v2 = cell[j];
                // consistent ordering to prevent double entries
                if (v1 < v2) {
                    vertices[v1].pairs.push(v2);
                } else {
                    vertices[v2].pairs.push(v1);
                }
            }
        });

        if (threshold > 0) {
            for (var i = 0; i < n; i++) {
                for (var j = i - 1; j >= 0; j--) {
                    if (vec3.distance(cells[i], cells[j]) < threshold) {
                        if (i < j) {
                            vertices[i].pairs.push(vertices[j]);
                        } else {
                            vertices[j].pairs.push(vertices[i]);
                        }
                    }
                }
            }
        }

        cells.map(function (cell, cellId) {
            var normal = faceNormals[cellId];
            // [a, b, c, d] where plane is defined by a*x+by+cz+d=0
            // choose the first vertex WLOG
            var pointOnTri = positions[cell[0]];
            var plane = [
                normal[0],
                normal[1],
                normal[2],
                -vec3.dot(normal, pointOnTri),
            ];

            cell.map(function (vertexId) {
                var errorQuadric = ndarray(new Float32Array(4 * 4), [4, 4]);
                for (var i = 0; i < 4; i++) {
                    for (var j = i; j >= 0; j--) {
                        var value = plane[i] * plane[j];
                        errorQuadric.set(i, j, value);
                        if (i != j) {
                            errorQuadric.set(j, i, value);
                        }
                    }
                }

                var existingQuadric = vertices[vertexId].error;
                ops.add(existingQuadric, existingQuadric, errorQuadric);
            });
        });

        var costs = new Heap(function (a, b) {
            return a.cost - b.cost;
        });

        var edges = [];
        vertices.map(function (v1) {
            v1.pairs.map(function (v2Index) {
                var v2 = vertices[v2Index];
                var optimal = optimalPosition(v1, v2);

                var edge = {
                    pair: [v1.index, v2Index],
                    cost: optimal.error,
                    optimalPosition: optimal.vertex,
                    optimalColor: optimal.color,
                };

                costs.push(edge);
                // to update costs
                edges.push(edge);
            });
        });

        var n = positions.length;
        return function (targetCount) {
            // deep-copy trick: https://stackoverflow.com/questions/597588/how-do-you-clone-an-array-of-objects-in-javascript
            var newCells = JSON.parse(JSON.stringify(cells));
            var deletedCount = 0;

            while (n - deletedCount > targetCount) {
                var leastCost = costs.pop();
                var i1 = leastCost.pair[0];
                var i2 = leastCost.pair[1];
                if (i1 == i2) {
                    // edge has already been collapsed
                    continue;
                }
                vertices[i1].position = leastCost.optimalPosition;
                vertices[i1].color = leastCost.optimalColor;

                for (var i = newCells.length - 1; i >= 0; i--) {
                    var cell = newCells[i];
                    var cellIndex2 = cell.indexOf(i2);
                    if (cellIndex2 != -1) {
                        if (cell.indexOf(i1) != -1) {
                            // Delete cells with zero area, as v1 == v2 now
                            newCells.splice(i, 1);
                        }

                        cell[cellIndex2] = i1;
                    }
                }

                var v1 = vertices[i1];
                edges.map(function (edge, i) {
                    var edgeIndex1 = edge.pair.indexOf(i1);
                    var edgeIndex2 = edge.pair.indexOf(i2);

                    if (edgeIndex1 != -1 && edgeIndex2 != -1) {
                        edge.pair[edgeIndex2] = i1;
                        return;
                    }

                    if (edge.pair.indexOf(i1) != -1) {
                        var optimal = optimalPosition(
                            v1,
                            vertices[edge.pair[(edgeIndex1 + 1) % 2]],
                        );
                        edge.optimalPosition = optimal.vertex;
                        edge.optimalColor = optimal.color;
                        edge.cost = optimal.error;
                    }

                    if (edge.pair.indexOf(i2) != -1) {
                        // use v1 as that is the new position of v2
                        var optimal = optimalPosition(
                            v1,
                            vertices[edge.pair[(edgeIndex2 + 1) % 2]],
                        );
                        edge.pair[edgeIndex2] = i1;
                        edge.optimalPosition = optimal.vertex;
                        edge.optimalColor = optimal.color;
                        edge.cost = optimal.error;
                    }
                });

                costs.heapify();
                deletedCount++;
            }

            return removeOrphans(
                newCells,
                vertices.map(function (p) {
                    return p.position;
                }),
                vertices.map(function (p) {
                    return p.color;
                }),
            );
        };
    }
    simplify = meshSimplify;
})();
