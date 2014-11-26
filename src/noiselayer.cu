#include <noiselayer.cuh>

// normalize so that each column represents probability using projection
void prob_project(NVMatrix& m) {
	const float eps = 0.000001;
	NVMatrix* pos = new NVMatrix(m);	// identify postive elements
	NVMatrix* x = new NVMatrix(m);
	pos->apply(NVMatrixOps::One());
	NVMatrix* r = new NVMatrix(m.getNumRows(), 1);
	NVMatrix* q = new NVMatrix(*r);
	while (true) {
		m.sum(1, *r);
		r->addScalar(-1);
		r->apply(NVMatrixOps::Abs(), *q);
		q->biggerThanScalar(eps);
		if (q->sum() ==  0) break;
		r->eltwiseDivide(pos->sum(1));
		m.addVector(*r, -1, *x);
		x->eltwiseMult(*pos);
		// pos->eltwiseMultByVector(*q);
		// x->eltwiseMultByVector(*q);
		pos->smallerThanScalar(0.5);
		m.eltwiseMult(*pos);
		m.add(*x);
		m.biggerThanScalar(0, *pos);
		// pos->smallerThanScalar(0.5);
		// pos->eltwiseMultByVector(*q);
		// pos->smallerThanScalar(0.5);
		m.eltwiseMult(*pos);
	}
	delete pos;
	delete x;
	delete r;
	delete q;
}		